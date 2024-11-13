import streamlit as st
import joblib
import numpy as np
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.transform import resize

# Load pre-trained models
pca_model_path = "PCA_ECG (1) (1).pkl"
prediction_model_path = "Heart_Disease_Prediction_using_ECG (4) (1).pkl"
pca_model = joblib.load(pca_model_path)
prediction_model = joblib.load(prediction_model_path)

# Title and description
st.title("Heart Disease Prediction Using ECG")
st.write("Upload ECG images to analyze and predict potential heart diseases.")

# File upload
uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "png"])

if uploaded_file:
    # Preprocessing the uploaded image
    image = imread(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert to grayscale and preprocess
    grayscale = rgb2gray(image)
    blurred_image = gaussian(grayscale, sigma=0.7)
    global_thresh = threshold_otsu(blurred_image)
    binary_global = blurred_image < global_thresh
    resized_image = resize(binary_global, (300, 450))
    
    st.image(resized_image, caption="Processed Image", use_column_width=True, clamp=True)

    # Flatten and apply PCA
    flattened_image = resized_image.flatten().reshape(1, -1)
    pca_features = pca_model.transform(flattened_image)

    # Predict using the loaded model
    prediction = prediction_model.predict(pca_features)
    prediction_label = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease Detected"

    # Display the result
    st.write(f"### Prediction: {prediction_label}")
