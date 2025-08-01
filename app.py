import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Iris Flower Predictor 🌸", layout="centered")

# 🌼 Custom background and styles
def set_bg():
    st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.pexels.com/photos/2471455/pexels-photo-2471455.jpeg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 1rem;
    }
    h1 {
        font-size: 2.4rem !important;
        font-weight: bold !important;
        color: #ffffff !important;
        text-align: left !important;
    }
    h3, label, .stButton>button {
        color: #ffffff !important;
        text-align: left !important;
    }
    </style>
    """, unsafe_allow_html=True)

set_bg()

# Load model
model = joblib.load("iris_rf_model.pkl")

# Species data
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

# Title and subtitle
st.markdown("### 🌸 Iris Species Predictor")
st.markdown("Enter the flower measurements below to predict the species.")

# Input fields
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

    st.markdown(f"""
        <div style='text-align: center; margin-top: 1rem;'>
            <img src='{info["image"]}' width='300' style='border: 3px solid white; border-radius: 10px;'>
            <p style='color: white; font-size: 16px;'>Iris {pred.capitalize()}</p>
        </div>
    """, unsafe_allow_html=True)

    st.write(f"📝 {info['desc']}")
