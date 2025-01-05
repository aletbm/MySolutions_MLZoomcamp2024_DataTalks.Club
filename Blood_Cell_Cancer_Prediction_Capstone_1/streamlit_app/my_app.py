import streamlit as st
import tensorflow as tf
from tensorflow.image import resize
import numpy as np
from io import BytesIO
from urllib import request as req
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2

def download_image(url):
    with req.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def preprocessing_image(url=None, array=None, size=None):
    if url:
        image = download_image(url)
        if image.mode != 'RGB':
            image = image.convert('RGB')
    else:
        image = array

    image = resize(image, size=size, method="nearest", antialias=True)
    return np.array([image], dtype=np.float32)/255.0

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

st.set_page_config(layout="wide")

st.write("""
# 🩸 Blood Cell Cancer Prediction - Classification & Segmentation
""")

model = tf.keras.models.load_model("/mount/src/mysolutions_mlzoomcamp2024_datatalks.club/Blood_Cell_Cancer_Prediction_Capstone_1/models/model_seg_clf.keras")

st.write("You have two options to load your image:")
st.write("#### First option")
url = st.text_input(
    "Enter your URL image here 👇",
    label_visibility="visible",
    disabled=False,
    placeholder="https://....",
    )
col1, col2, col3 = st.columns(3)

if url:
    image = preprocessing_image(url=url, size=(192, 256))
    with col1:
        st.header("Input image")
        st.image(image[0], caption="Preprocessed image ")
    
    pred = model.predict(image)
    class_ = pred["classification"]
    class_ = ['[Malignant] early Pre-B', '[Malignant] Pro-B', '[Malignant] Pre-B', 'Benign'][np.argmax(class_)]
    mask = pred["segmentation"]
    
    with col2:
        st.header("Segmentation")
        st.image(mask, caption="Predicted mask")
        st.header("Classification")
        st.write(f"Predicted class: {class_}")
    
    with col3:
        image = np.array(image[0]*255, dtype=np.uint8)
        mask = np.array(np.round(mask[0]), dtype=np.uint8)
        image_cropped = apply_mask(image, mask)
        st.header("Applied mask")
        st.image(image_cropped, caption="Original cropped image")

st.write("#### Second option")
uploaded_file = st.file_uploader("Choose a file")
col1, col2, col3 = st.columns(3)
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    image = np.array(Image.open(BytesIO(bytes_data)).convert('RGB'))
    image = preprocessing_image(array=image, size=(192, 256))
    with col1:
        st.header("Input image")
        st.image(image[0], caption="Preprocessed image ")
    
    pred = model.predict(image)
    class_ = pred["classification"]
    class_ = ['[Malignant] early Pre-B', '[Malignant] Pro-B', '[Malignant] Pre-B', 'Benign'][np.argmax(class_)]
    mask = pred["segmentation"]
    
    with col2:
        st.header("Segmentation")
        st.image(mask, caption="Predicted mask")
        st.header("Classification")
        st.write(f"Predicted class: {class_}")
    
    with col3:
        image = np.array(image[0]*255, dtype=np.uint8)
        mask = np.array(np.round(mask[0]), dtype=np.uint8)
        image_cropped = apply_mask(image, mask)
        st.header("Applied mask")
        st.image(image_cropped, caption="Original cropped image")
#streamlit run ./app/my_app.py
#https://i.postimg.cc/c1Q236XF/0-6114.jpg
#https://i.postimg.cc/fVbbm2g1/2.png