# app.py
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
        background-image: url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 1rem;
    }
    h1, h3, label, .stButton>button {
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_bg()

# Load model
model = joblib.load("iris_rf_model.pkl")

# Species info
species_info = {
    "setosa": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_setosa_2.jpg",
        "desc": "Iris Setosa has small, purple-blue flowers and usually grows in cooler regions."
    },
    "versicolor": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
        "desc": "Iris Versicolor, or Blue Flag, is known for its violet-blue petals and wetland habitat."
    },
    "virginica": {
        "image": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
        "desc": "Iris Virginica produces larger flowers and thrives in marshy areas of North America."
    }
}

# App title
st.markdown("<h1 style='text-align: center;'>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Enter the flower measurements to predict the species</h3>", unsafe_allow_html=True)

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
