# app.py
import os
import base64
from pathlib import Path
from io import BytesIO

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# -------------------- CONFIG --------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# -------------------- MODEL --------------------
class MedicalImageClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.features = nn.Sequential(*list(self.backbone.children())[:-2])
        self.avgpool = self.backbone.avgpool
        self.classifier = self.backbone.fc

    def forward(self, x):
        feat = self.features(x)
        pooled = self.avgpool(feat)
        pooled = torch.flatten(pooled, 1)
        return self.classifier(pooled)

# -------------------- LOAD CHECKPOINT --------------------
checkpoint_path = "best_model.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"{checkpoint_path} not found!")

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model = MedicalImageClassifier(num_classes=checkpoint.get("num_classes", 2))
model.load_state_dict(checkpoint["model_state_dict"])
CLASS_NAMES = checkpoint.get("class_names", ["Class0", "Class1"])
model.to(DEVICE).eval()

# -------------------- GRADCAM --------------------
class GradCAMVisualizer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        gradients = self.gradients[0]  # [C,H,W]
        activations = self.activations[0]  # [C,H,W]

        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        cam = torch.zeros(activations.shape[1:], device=DEVICE)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def visualize_cam(self, input_tensor, target_class, original_image, alpha=0.4):
        cam = self.generate_cam(input_tensor, target_class)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Convert original image to numpy array
        if isinstance(original_image, Image.Image):
            orig = np.array(original_image.resize((224, 224)))
        else:
            orig = original_image

        overlay = cv2.addWeighted(heatmap, alpha, orig, 1 - alpha, 0)
        return cam_resized, heatmap, overlay

# Use last conv layer of ResNet18
cam = GradCAMVisualizer(model, model.backbone.layer4[-1].conv2)

# -------------------- UTILS --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image_as_tensor(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

# -------------------- FLASK APP --------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        f = request.files["file"]
        if f.filename == "":
            return jsonify({"success": False, "error": "Empty filename"}), 400
        if not allowed_file(f.filename):
            return jsonify({"success": False, "error": "Unsupported file type"}), 400

        # Save
        safe_name = secure_filename(f.filename)
        save_path = UPLOAD_DIR / safe_name
        f.save(str(save_path))

        # Preprocess
        x = load_image_as_tensor(save_path).to(DEVICE)

        # Prediction
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            pred_conf = float(probs[pred_idx])

        # Grad-CAM
        heat_b64 = None
        try:
            _, _, overlay = cam.visualize_cam(x, pred_idx, Image.open(save_path))
            _, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            heat_b64 = base64.b64encode(buf).decode("utf-8")
        except Exception as e:
            print("Grad-CAM Error:", e)

        return jsonify({
            "success": True,
            "filename": safe_name,
            "predicted_class": pred_idx,
            "predicted_label": CLASS_NAMES[pred_idx],
            "confidence": pred_conf,
            "gradcam_overlay": heat_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
