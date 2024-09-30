# OCR for Hindi and English - README

## Overview

This project implements Optical Character Recognition (OCR) for both Hindi and English languages. It uses a pre-trained VisionEncoderDecoderModel for Hindi OCR and Microsoft's TrOCR model for English OCR. The application is deployed via **Streamlit**, allowing users to upload images and get text extracted in their selected language (Hindi or English).

### Features:
- **Hindi OCR**: Using a custom-trained VisionEncoderDecoderModel for handwritten/printed Hindi text.
- **English OCR**: Using the pre-trained Microsoft TrOCR model for handwritten English text.
- **Image Upload**: Users can upload images in `jpg`, `jpeg`, or `png` formats.
- **Language Selection**: Users can select between Hindi or English for OCR.

---

## Setup Instructions

### 1. Prerequisites
Make sure you have the following installed:
- Python 3.7+
- CUDA (Optional, for GPU support)

### 2. Install Required Libraries

Run the following command to install the required dependencies:

```bash
pip install torch torchvision transformers numpy opencv-python-headless Pillow streamlit
```

### 3. Download Model Checkpoints

1. **Download the checkpoints**: 
   Download the pre-trained model files for Hindi OCR from the provided source or your Google Drive and ensure you have the following directories:
   - `checkpoint-56000` folder: Contains the pre-trained weights for the VisionEncoderDecoderModel.
   - `hindi_model` folder: Contains additional model-specific files if needed.

2. **Replace the paths**:
   After downloading, replace the paths in the code with the local paths where you store these models on your machine. You can do this by updating the `checkpoint_dir` and `hindi_model_path` variables in the code.

Example:
```python
checkpoint_dir = "path_to_your_local_directory/checkpoint-56000"
hindi_model_path = "path_to_your_local_directory/hindi_model"
```

### 4. Running the Application

1. **Start Streamlit**:
   Run the following command to start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. **Upload Images**:
   After the app starts, you can upload an image (in `jpg`, `jpeg`, or `png` format) and select the OCR language (Hindi or English). Then, click the **"Perform OCR"** button to get the predicted text.

---

## Detailed Code Explanation

### Imports
- **torch**: Used for handling deep learning models, such as VisionEncoderDecoderModel.
- **numpy**: For numerical computations and handling image data.
- **cv2 (OpenCV)**: Used for image manipulation and loading image files.
- **PIL (Pillow)**: To handle image formats.
- **transformers**: Provides models and tokenizers from Hugging Face's library.
- **streamlit**: A framework for building web applications easily.

### Model and Tokenizer
- **Hindi Model**: The VisionEncoderDecoderModel is loaded from the pre-trained `checkpoint-56000` directory. The tokenizer is also loaded from this directory to convert the predicted model output into text.
- **English Model**: The pre-trained `microsoft/trocr-base-handwritten` model is used for English OCR.

### Image Transformation
Before sending images to the model, they are transformed:
- Resized to `(384x384)`
- Converted to tensor format
- Normalized using ImageNet normalization values

### OCR Prediction Functions
- **`predict_hindi_ocr(img)`**: Converts the image to a tensor, performs inference using the Hindi OCR model, and decodes the result into a readable text.
- **`english_ocr(img)`**: Uses the pipeline function to perform OCR on the image using the pre-trained TrOCR model.

### Streamlit Web Application
- **Title**: Displays a title for the app.
- **File Uploader**: Lets the user upload an image for OCR.
- **Language Selection**: Lets the user choose between Hindi and English OCR.
- **Perform OCR Button**: After clicking, it processes the uploaded image and displays the predicted text.

### Final Output
The predicted text is displayed below the OCR button once the image is processed.

---

## Usage Example

1. **Step 1**: Start the application by running `streamlit run app.py`.
2. **Step 2**: Upload an image in the supported format.
3. **Step 3**: Select the language for OCR (Hindi or English).
4. **Step 4**: Click the "Perform OCR" button to get the predicted text from the image.

---

## Notes
- Ensure your machine has enough resources (CPU/GPU) for processing, especially when dealing with large images or models.
- If you're using a GPU, CUDA support will be automatically enabled if available.
- You can modify the `transform` function to adapt image pre-processing to your specific requirements.

---

## Troubleshooting
1. **Module Not Found Errors**: Ensure that all required Python packages are installed.
2. **Model Loading Issues**: Verify that the model files are correctly downloaded and that the paths in the code are correct.
3. **CUDA Not Available**: If using a GPU and encountering CUDA errors, ensure that you have a compatible version of PyTorch installed with CUDA support.

