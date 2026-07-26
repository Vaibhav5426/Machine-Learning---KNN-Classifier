import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Red Wine Quality Predictor",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_model():
    with open('wine_knn_model.pkl', 'rb') as file:
        data = pickle.load(file)

    # Check if loaded data is a dictionary containing expected keys
    if isinstance(data, dict):
        return data['model'], data['scaler']
    else:
        st.error(
            "The pickle file contains an unbundled object instead of a model dictionary. "
            'Please re-save using the dictionary structure.'
        )
        st.stop()
# --- HEADER SECTION ---
st.title("🍷 Red Wine Quality Predictor")
st.markdown("""
Welcome to the AI Sommelier! Adjust the chemical properties below to predict whether a red wine is of **Premium Quality (Score $\ge$ 7)** or **Average/Standard**.
""")
st.markdown("---")

# --- USER INPUT SECTION ---
st.header("🧪 Chemical Properties")

# Row 1: Acidity and Minerals
col1, col2 = st.columns(2)
with col1:
    st.subheader("Acidity & pH")
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 8.3)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.5)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.25)
    pH = st.slider("pH Level", 2.7, 4.0, 3.3)

with col2:
    st.subheader("Minerals & Sugar")
    residual_sugar = st.slider("Residual Sugar", 0.5, 15.5, 2.5)
    chlorides = st.slider("Chlorides", 0.01, 0.6, 0.08)
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.65)
    alcohol = st.slider("Alcohol (%)", 8.0, 15.0, 10.4)

st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Sulfur and Density (Using number inputs for precision)
st.subheader("Sulfur & Density")
col3, col4, col5 = st.columns(3)
with col3:
    free_sulfur_dioxide = st.number_input("Free SO2", min_value=1.0, max_value=75.0, value=14.0)
with col4:
    total_sulfur_dioxide = st.number_input("Total SO2", min_value=6.0, max_value=290.0, value=46.0)
with col5:
    density = st.number_input("Density", min_value=0.9900, max_value=1.0040, value=0.9960, format="%.4f")

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("🔮 Predict Wine Quality", type="primary", use_container_width=True):
    
    # Bundle inputs exactly in the order the model was trained on
    input_features = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, 
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, 
        density, pH, sulphates, alcohol
    ]])
    
    # Scale features using the loaded StandardScaler
    features_scaled = scaler.transform(input_features)
    
    # Make Prediction
    prediction = knn.predict(features_scaled)[0]
    probability = knn.predict_proba(features_scaled)[0][1]
    
    # --- DISPLAY RESULTS ---
    st.header("Results")
    
    if prediction == 1:
        st.success("🌟 **Great news!** This chemical profile indicates a **Premium Quality** wine.")
        st.metric(label="Model Confidence", value=f"{probability * 100:.1f}%")
        st.balloons()
    else:
        st.warning("📉 **Standard Quality.** This chemical profile indicates an **Average or Below Average** wine.")
        st.metric(label="Probability of being Premium", value=f"{probability * 100:.1f}%")
        
    # Visual confidence bar
    st.progress(float(probability), text="Premium Quality Probability")
