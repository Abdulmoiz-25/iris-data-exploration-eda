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
        filter: blur(8px); /* Blur the background image */
    }
    /* Main content container styling */
    div[data-testid="stAppViewContainer"] > section {
        background-color: rgba(0, 0, 0, 0.7); /* Dark overlay for content */
        filter: none; /* Ensure content is not blurred */
        padding: 2rem;
        border-radius: 1rem;
    }
    /* Ensure all text is white and not blurred */
    h1, h2, h3, p, label, div, span {
        color: #ffffff !important;
        filter: none !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    /* Button styling */
    .stButton > button {
        color: #ffffff !important;
        background-color: rgba(0, 0, 0, 0.5) !important;
        border: 1px solid #ffffff !important;
        width: 100%; /* Make buttons fill their column */
        height: 38px; /* Match height of number input */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    /* Number input styling to look like a bar */
    div[data-testid="stNumberInput"] > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 0.5rem !important;
        color: #ffffff !important;
        padding: 0.5rem 1rem !important;
        text-align: center !important;
        font-size: 1.1rem !important;
    }
    /* Hide default number input arrows */
    div[data-testid="stNumberInput"] input::-webkit-outer-spin-button,
    div[data-testid="stNumberInput"] input::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    div[data-testid="stNumberInput"] input[type=number] {
        -moz-appearance: textfield;
    }
    /* Custom value limit display */
    .value-limits {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-top: 5px;
        color: #ccc;
    }
    </style>
    """, unsafe_allow_html=True)

set_bg()

# Load model
# Make sure 'iris_rf_model.pkl' is in the same directory as app.py
try:
    model = joblib.load("iris_rf_model.pkl")
except FileNotFoundError:
    st.error("Model file 'iris_rf_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop() # Stop execution if model is not found

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

# Function to create a custom number input with +/- buttons and limits
def custom_number_input(label, min_value, max_value, default_value, step, key):
    st.write(f"**{label}**")
    
    # Initialize session state for the value if not already set
    if f'{key}_value' not in st.session_state:
        st.session_state[f'{key}_value'] = default_value

    cols = st.columns([0.15, 0.7, 0.15]) # Adjust column ratios for buttons and input

    with cols[0]:
        if st.button("-", key=f"{key}_minus"):
            st.session_state[f'{key}_value'] = max(min_value, st.session_state[f'{key}_value'] - step)
    
    with cols[1]:
        # Use st.number_input to display and allow direct typing
        current_value = st.number_input(
            " ", # Label is empty as it's handled by st.write above
            min_value=min_value,
            max_value=max_value,
            value=st.session_state[f'{key}_value'],
            step=step,
            key=f"{key}_input",
            label_visibility="collapsed" # Hide default label
        )
        # Update session state if user types directly into the number input
        st.session_state[f'{key}_value'] = current_value

    with cols[2]:
        if st.button("+", key=f"{key}_plus"):
            st.session_state[f'{key}_value'] = min(max_value, st.session_state[f'{key}_value'] + step)
    
    # Display min/max limits below the input
    st.markdown(f"<div class='value-limits'><span>Min: {min_value}</span><span>Max: {max_value}</span></div>", unsafe_allow_html=True)
    
    return st.session_state[f'{key}_value']

# App title
st.markdown("<h1>🌸 Iris Species Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter the flower measurements to predict the species</h3>", unsafe_allow_html=True)

# Input fields using the custom function
sepal_length = custom_number_input("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1, "sepal_length")
sepal_width = custom_number_input("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1, "sepal_width")
petal_length = custom_number_input("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1, "petal_length")
petal_width = custom_number_input("Petal Width (cm)", 0.1, 2.5, 0.1, "petal_width")

# Prediction
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("🔍 Predict Species"):
    pred = model.predict(features)[0]
    info = species_info[pred]
    st.success(f"🌼 Predicted Species: **{pred.capitalize()}**")
    st.image(info["image"], caption=f"Iris {pred.capitalize()}", use_column_width=True)
    st.write(f"📝 {info['desc']}")
