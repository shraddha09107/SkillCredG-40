# AI-Powered Pneumonia Detection from Chest X-rays

## Overview

This project implements an AI-powered system for detecting pneumonia from chest X-ray images. It uses a deep learning model trained on a large dataset of X-ray images to provide accurate diagnoses and visual explanations, assisting medical professionals in identifying pneumonia.

## Demo

[Link to Deployed Application] (Will be added after deployment)

## Key Features

-   **AI-powered Pneumonia Detection:** Utilizes a deep learning model for accurate pneumonia detection in chest X-ray images.
-   **Grad-CAM Visualization:** Provides visual explanations highlighting the areas of the X-ray image the AI focused on for its diagnosis, enhancing transparency and trust.
-   **User-Friendly Web Interface:** A clean and intuitive web interface allows easy image uploads and result interpretation.
-   **REST API:** A well-documented API enabling integration with other medical systems.

## Tech Stack

-   **Python:** The primary programming language.
-   **Flask:** A micro web framework for building the API and web interface.
-   **PyTorch:** An open-source machine learning framework used for model training and inference.
-   **Torchvision:** PyTorch's library for computer vision tasks, including image transformations and model architectures.
-   **scikit-learn:** Machine learning library.
-   **opencv-python**: Image processing library.
-   **numpy**: For numerical computations.
-   **PIL (Pillow):** Python Imaging Library for image handling.
-   **HTML, CSS, JavaScript:** For building the web interface.
-   **Bootstrap:** A CSS framework for styling the web interface.
-   **gunicorn**: WSGI server.

## Setup Instructions

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [repository_url]  <- Replace with your repository URL
    cd [repository_name]       <- Replace with your repository name
    ```

2.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    ```

3.  **Activate the virtual environment:**

    *   On macOS and Linux:

        ```bash
        source .venv/bin/activate
        ```

    *   On Windows:

        ```bash
        .venv\Scripts\activate
        ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Place the model**: Confirm that `best_model.pth` is placed in the same directory where `app.py` is.

6.  **Run the application:**

    ```bash
    python app.py
    ```

7.  **Open your web browser:** Navigate to `http://127.0.0.1:5000/` to access the application.

## Usage Instructions

1.  **Upload an X-ray image:** Use the file upload form to select a chest X-ray image from your computer. Supported formats include JPEG, PNG, and DICOM.
2.  **View the diagnosis:** The AI model will analyze the image and display the prediction results, including the probability of pneumonia being present.
3.  **Interpret the Grad-CAM visualization:** The Grad-CAM visualization will highlight the regions of the X-ray image that the AI focused on when making its diagnosis. Brighter areas indicate higher attention.

## Model Training

The deep learning model was trained using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.
[Describe your training process briefly. What were the main steps? What were the main challenges?]
The training process was performed in Google Colab using a ResNet18 architecture. See the `Colab_Notebook.ipynb` notebook for details on the training procedure.

## Future Improvements

-   Support for more image formats (e.g., DICOM).
-   Improved Grad-CAM visualizations (e.g., higher resolution, better color schemes).
-   Integration with medical record systems.
-   Deployment to a cloud platform for wider accessibility.
-   Implementation of unit tests for increased code reliability.
-   Incorporate user feedback to improve model accuracy and usability.

## Contributions
[Did anyone else contribute? Put names here.]

## Disclaimer

This AI tool is designed for educational and research purposes only. It should not replace professional medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical advice and diagnosis.

## License

[Choose a license. Options include MIT, Apache 2.0, or GPL 3.0]

* **MIT License:** A permissive license that lets others use, modify, and distribute your code, even for commercial purposes, as long as they include the original copyright and license notice in any copy of your code/software.
* **Apache 2.0 License:** Similar to MIT but also grants patent rights.
* **GPL 3.0 License:** A strong copyleft license that requires derivative works to be licensed under GPL as well.
* **If you choose to keep the code to yourself, you can just note 'All rights reserved.**

## Credits

*   This project utilizes the following open-source libraries:
    *   Flask
    *   PyTorch
    *   Torchvision
    *   scikit-learn
    *   opencv-python
    *   numpy
    *   Pillow
    *   Bootstrap
