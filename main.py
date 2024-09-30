import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as v2
from transformers import VisionEncoderDecoderModel, AutoTokenizer, pipeline
import streamlit as st


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check for GPU


checkpoint_dir = "D:/User/Documents/Heytt Csbs/Python/checkpoint-56000"
hindi_model_path = "D:/User/Documents/Heytt Csbs/Python/hindi_model"
  


hindi_model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir, ignore_mismatched_sizes=True).to(device)
hindi_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)


english_ocr = pipeline('image-to-text', model="microsoft/trocr-base-handwritten")


transform = v2.Compose([
    v2.Resize((384, 384)),
    v2.ToTensor(),
    v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def predict_hindi_ocr(img):
    img_t = transform(img).unsqueeze(0).to(device)  
    hindi_model.eval()  
    with torch.no_grad():  
        generated_ids = hindi_model.generate(img_t)
    generated_text = hindi_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


st.title("OCR for Hindi and English")


uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])


language = st.selectbox("Select language for OCR", ("Hindi", "English"))


if st.button('Perform OCR'):
    if uploaded_file is not None:
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  

        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        pil_img = Image.fromarray(img_rgb) 

        
        if language == "English":
          predicted_text = english_ocr(pil_img)[0]['generated_text']
            #predicted_text = predict_hindi_ocr(pil_img)  
        else:
           
            predicted_text = predict_hindi_ocr(pil_img)

        #
        st.write(f"Predicted text: {predicted_text}")
    else:
        st.write("Please upload an image.")
