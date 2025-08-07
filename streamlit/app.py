import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Judul aplikasi
st.title("Water Quality Prediction")

# Load model dan scaler
model = pickle.load('model.pkl')
scaler = pickle.load('scaler.pkl')  # kalau pakai scaler

st.title("Prediksi Kualitas Air Berdasarkan Parameter")

# Input dari user
water_temp = st.number_input("Water Temperature (°C)", 0.0, 50.0, 25.0)
ph = st.number_input("pH", 0.0, 14.0, 7.0)
dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 8.0)
conductivity = st.number_input("Conductivity (μS/cm)", 0.0, 2000.0, 500.0)
nitrate = st.number_input("Nitrate (mg/L)", 0.0, 100.0, 10.0)
total_phosphorus = st.number_input("Total Phosphorus (mg/L)", 0.0, 5.0, 0.1)
total_nitrogen = st.number_input("Total Nitrogen (mg/L)", 0.0, 50.0, 5.0)
cod = st.number_input("COD (mg/L)", 0.0, 200.0, 20.0)
bod = st.number_input("BOD (mg/L)", 0.0, 100.0, 10.0)
heavy_metal_pb = st.number_input("Heavy Metals Pb (μg/L)", 0.0, 100.0, 10.0)

# Prediksi
if st.button("Prediksi Kualitas Air"):
    if model is None:
        st.error("Model belum tersedia. Silakan training model terlebih dahulu.")
    else:
        # Buat data input
        input_data = np.array([[
            water_temp, ph, dissolved_oxygen, conductivity, nitrate,
            total_phosphorus, total_nitrogen, cod, bod, heavy_metal_pb
        ]])
        
        # Scale input
        input_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Tampilkan hasil
        st.header("Hasil Prediksi")
        
        if prediction == 1:
            st.success("**Kualitas Air: BAIK/SANGAT BAIK**")
        else:
            st.error("**Kualitas Air: BURUK/SEDANG**")
        
        # Tampilkan probabilitas
        st.write(f"**Confidence:** {max(probability):.2%}")