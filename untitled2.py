# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 16:29:42 2025

@author: Irshad Alam
"""


import streamlit as st
import pickle
import numpy as np
import pandas as pd


import os
print(os.getcwd())

# predict_app.py

# Use full path
model_path = 'D:/Machine Learning/House Pricing Prediction/savemodel/models.pkl'
scaler_path = 'D:/Machine Learning/House Pricing Prediction/savemodel/scalers.pkl'

#df = pd.read_csv('D:/Machine Learning/House Pricing Prediction/dataset/Housing.csv') # If local, or specify full path as needed

# Load model and scaler (new files)
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("🏠 House Price Prediction App")
st.markdown("Enter house details to estimate the selling price.")

# Input fields
area = st.number_input("Area (sq ft)", value=1000)
bedrooms = st.slider("Bedrooms", 1, 6, 3)
bathrooms = st.slider("Bathrooms", 1, 4, 2)
stories = st.slider("Stories", 1, 4, 2)
parking = st.selectbox("Parking (cars)", [0, 1, 2, 3])

mainroad = st.radio("Is it on the main road?", ["yes", "no"])
guestroom = st.radio("Guest room available?", ["yes", "no"])
basement = st.radio("Basement available?", ["yes", "no"])
hotwater = st.radio("Hot water heating?", ["yes", "no"])
ac = st.radio("Air conditioning?", ["yes", "no"])
prefarea = st.radio("Preferred area?", ["yes", "no"])

furnishing = st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])

# Encoding binary and categorical inputs
# Correct furnishing encoding: only one dummy column used during training
furnishing_semi = 1 if furnishing == "semi-furnished" else 0  # 'semi-furnished' dummy column

# Input in correct order (12 features expected by the scaler/model)
input_data = [
    area, bedrooms, bathrooms, stories, parking,
    1 if mainroad == "yes" else 0,
    1 if guestroom == "yes" else 0,
    1 if basement == "yes" else 0,
    1 if hotwater == "yes" else 0,
    1 if ac == "yes" else 0,
    1 if prefarea == "yes" else 0,
    furnishing_semi
]



# Predict button
if st.button("Predict Price"):
    input_array = np.array([input_data])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    predicted_price = prediction[0]  # Get float value

    # If model was trained in lakhs
    st.success(f"🏡 Estimated House Price: ₹ {predicted_price * 1_00_00_000:,.2f}")



st.write(" Input Features (raw):", input_data)
st.write(" Scaled Features:", scaler.transform([input_data]))
scaled_input = scaler.transform([input_data])
prediction = model.predict(scaled_input)
