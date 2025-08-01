import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Iris Flower Predictor 🌸", layout="centered")

# Custom background using CSS
def set_bg():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/7284/flowers-garden.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
        filter: blur(8px);
    }
    .stApp > div {
        background-color: rgba(0, 0, 0, 0.7);
        filter: none;
    }
    div[data-testid="stAppViewContainer"] {
        background-color: rgba(0, 0, 0, 0.7);
        filter: none;
    }
    h1, h2, h3, p, label, div {
        color: #ffffff !important;
        filter: none !important;
    }
    .stButton > button {
        color: #ffffff !important;
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid #ffffff !important;
    }
    /* Slider styling to look like bars */
    .stSlider > div > div > div > div {
        background-color: rgba(255, 255, 255, 0.2) !important;
        height: 8px !important;
        border-radius: 4px !important;
    }

    .stSlider > div > div > div > div > div {
        background-color: #4CAF50 !important;
        height: 8px !important;
        border-radius: 4px !important;
    }

    .stSlider > div > div > div > div > div > div {
        background-color: #ffffff !important;
        border: 2px solid #4CAF50 !important;
        width: 20px !important;
        height: 20px !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
    }

    /* Show min/max values */
    .stSlider > div > div > div {
        position: relative !important;
    }

    .stSlider > div > div > div::before {
        content: attr(data-min) !important;
        position: absolute !important;
        left: 0 !important;
        bottom: -20px !important;
        color: #ffffff !important;
        font-size: 12px !important;
    }

    .stSlider > div > div > div::after {
        content: attr(data-max) !important;
        position: absolute !important;
        right: 0 !important;
        bottom: -20px !important;
        color: #ffffff !important;
        font-size: 12px !important;
    }

    /* Slider labels */
    .stSlider > label {
        color: #ffffff !important;
        font-weight: bold !important;
        margin-bottom: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_bg()

# Load model
model = joblib.load("iris_rf_model.pkl")

# Species info
species_info = {
    "setosa": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_setosa.JPG/640px-Iris_setosa.JPG",
        "desc": "Iris Setosa has small, purple-blue flowers and usually grows in cooler regions."
    },
    "versicolor": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Iris_versicolor_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-073.jpg/640px-Iris_versicolor_-_K%C3%B6hler%E2%80%93s_Medizinal-Pflanzen-073.jpg",
        "desc": "Iris Versicolor, or Blue Flag, is known for its violet-blue petals and wetland habitat."
    },
    "virginica": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Iris_virginica.jpg/640px-Iris_virginica.jpg",
        "desc": "Iris Virginica produces larger flowers and thrives in marshy areas of North America."
    }
}

# App title
st.markdown("<h1>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter the flower measurements to predict the species</h3>", unsafe_allow_html=True)

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prediction
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("🔍 Predict Species"):
    pred = model.predict(features)[0]
    info = species_info[pred]
    st.success(f"🌼 Predicted Species: **{pred.capitalize()}**")
    st.image(info["image"], caption=f"Iris {pred.capitalize()}", use_column_width=True)
    st.write(f"📝 {info['desc']}")
