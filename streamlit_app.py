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
st.markdown("An academic and professional decision-support tool for predicting weekly SCFI market movements using a Keras Slim Residual Neural Network.")

st.header("Input Variables")

# Opprett to kolonner for inputfeltene
col1, col2 = st.columns(2)

with col1:
    scfi = st.number_input("SCFI", value=1500.0)
    lag1 = st.number_input("Lag 1 (SCFI value one week ago)", value=1480.0)
    lag2 = st.number_input("Lag 2 (SCFI value two weeks ago)", value=1460.0)
    lag3 = st.number_input("Lag 3 (SCFI value three weeks ago)", value=1440.0)
    lag4 = st.number_input("Lag 4 (SCFI value four weeks ago)", value=1420.0)

with col2:
    timecharter = st.number_input("Timecharter Index", value=120.0)
    idle_teu = st.number_input("Idle TEU", value=1.5)
    secondhand = st.number_input("Secondhand Index", value=75.0)
    china_port = st.number_input("China Port Calls TEU", value=9000000.0)

# Automatically generate engineered features
trend1 = (scfi - lag1) / lag1
trend2 = (scfi - lag2) / lag2
trend3 = (scfi - lag3) / lag3
trend4 = (scfi - lag4) / lag4

moving_average = np.mean([scfi, lag1, lag2, lag3, lag4])

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
