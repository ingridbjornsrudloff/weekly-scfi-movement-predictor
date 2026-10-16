import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Sett sidetittel
st.set_page_config(page_title="SCFI Market Predictor", layout="centered")

# Last inn modell og preprocessors
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('keras_scfi_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return model, scaler, encoder

try:
    model, scaler, encoder = load_resources()
    st.success("Modell og skalerere lastet inn suksessfullt!")
except Exception as e:
    st.error(f"Feil under lasting av filer: {e}")

st.title("📈 SCFI Weekly Market Movement Predictor")
st.markdown("An academic and professional tool to forecast weekly Shanghai Containerized Freight Index (SCFI) trends using a Keras Slim Residual Neural Network.")

st.header("Input Variables")

# Opprett to kolonner for inputfeltene
col1, col2 = st.columns(2)

with col1:
    scfi = st.number_input("SCFI", value=1500.0)
    lag1 = st.number_input("Lag 1", value=1480.0)
    lag2 = st.number_input("Lag 2", value=1460.0)
    lag3 = st.number_input("Lag 3", value=1440.0)
    lag4 = st.number_input("Lag 4", value=1420.0)
    moving_average = st.number_input("Moving Average", value=1450.0)
    trend1 = st.number_input("Trend 1", value=0.02, format="%.4f")

with col2:
    trend2 = st.number_input("Trend 2", value=0.01, format="%.4f")
    trend3 = st.number_input("Trend 3", value=-0.01, format="%.4f")
    trend4 = st.number_input("Trend 4", value=0.005, format="%.4f")
    timecharter = st.number_input("Timecharter Index", value=120.0)
    idle_teu = st.number_input("Idle TEU", value=1.5)
    secondhand = st.number_input("Secondhand Index", value=75.0)
    china_port = st.number_input("China Port Calls TEU", value=9000000.0)

if st.button("Generate Prediction", type="primary"):
    # Samle alle features i riktig rekkefølge
    input_data = np.array([[
        scfi, lag1, lag2, lag3, lag4, moving_average, 
        trend1, trend2, trend3, trend4, timecharter, 
        idle_teu, secondhand, china_port
    ]])
    
    # Skaler inndataene
    input_scaled = scaler.transform(input_data)
    
    # Gjør prediksjon
    probs = model.predict(input_scaled)[0]
    
    # Hent klasser
    classes = encoder.classes_
    predicted_class = classes[np.argmax(probs)]
    
    st.subheader("Results")
    
    # Vis predikert klasse med fargeindikator
    if predicted_class == 'Up':
        st.success(f"### Predicted Market Movement: **{predicted_class}** 📈")
    elif predicted_class == 'Down':
        st.error(f"### Predicted Market Movement: **{predicted_class}** 📉")
    else:
        st.info(f"### Predicted Market Movement: **{predicted_class}** ➡️")
        
    # Vis sannsynlighetsfordeling
    st.write("**Class Probabilities:**")
    for c, p in zip(classes, probs):
        st.write(f"- **{c}**: {p*100:.2f}%")
        st.progress(float(p))
