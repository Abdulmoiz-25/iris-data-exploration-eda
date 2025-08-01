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
        background-position: center;
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        backdrop-filter: blur(8px);
        z-index: -1;
    }
    .stApp > div {
        background-color: rgba(0, 0, 0, 0.7);
    }
    h1, h2, h3, p, label, div {
        color: #ffffff !important;
    }
    .stButton > button {
        color: #ffffff !important;
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid #ffffff !important;
    }
    /* Purple slider thumb */
    input[type="range"]::-webkit-slider-thumb {
        background: #6f42c1 !important; /* Added !important */
        border: 2px solid #ffffff !important; /* Added !important */
    }
    input[type="range"]::-moz-range-thumb {
        background: #6f42c1 !important; /* Added !important */
        border: 2px solid #ffffff !important; /* Added !important */
    }
    input[type="range"]::-ms-thumb {
        background: #6f42c1 !important; /* Added !important */
        border: 2px solid #ffffff !important; /* Added !important */
    }
    </style>
    """, unsafe_allow_html=True)

set_bg()

# Load the trained model
model = joblib.load("iris_rf_model.pkl")

# Local species image paths (relative to your repo)
species_info = {
    "setosa": {
        "image": "images/Iris setosa.jpg",
        "desc": "Iris Setosa has small, purple-blue flowers and usually grows in cooler regions."
    },
    "versicolor": {
        "image": "images/Blue_Flag,_Ottawa.jpg",
        "desc": "Iris Versicolor, or Blue Flag, is known for its violet-blue petals and wetland habitat."
    },
    "virginica": {
        "image": "images/Iris_virginica.jpg",
        "desc": "Iris Virginica produces larger flowers and thrives in marshy areas of North America."
    }
}

# App header
st.markdown("<h1 style='font-size: 42px;'>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
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
    st.image(info["image"], caption=f"Iris {pred.capitalize()}", use_container_width=True)
    st.write(f"📝 {info['desc']}")
