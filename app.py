import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

#  Load Trained Model
model = keras.models.load_model('digits_model.h5')

#  Streamlit Page Config
st.set_page_config(page_title="Handwritten Digit Classifier", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>🧠 Handwritten Digit Classifier</h1>", unsafe_allow_html=True)
st.write("Upload an image of a digit (**28x28**, white digit on black background preferred)")

#  Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show Uploaded Image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", width=150)

    #  Preprocess Image
    img = ImageOps.invert(image)  # Invert: white on black
    img = img.resize((28, 28))  # Resize to MNIST format
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 784)  # Flatten

    #  Make Prediction
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    #  Show Result
    st.markdown(f"<h3>🎯 Predicted Digit: <span style='color: lightgreen;'>{predicted_digit}</span></h3>", unsafe_allow_html=True)
