import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAMVisualizer:
    """Grad-CAM for ResNet18 backbone (custom MedicalImageClassifier)"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

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

        cam = torch.zeros(activations.shape[1:])
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.cpu().numpy()

    def visualize_cam(self, input_tensor, target_class, original_image, alpha=0.4):
        cam = self.generate_cam(input_tensor, target_class)
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Convert original image to numpy
        import numpy as np
        if hasattr(original_image, "size"):
            orig = np.array(original_image.resize((224, 224)))
        else:
            orig = original_image

        overlay = heatmap * alpha + orig * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return cam_resized, heatmap, overlay
