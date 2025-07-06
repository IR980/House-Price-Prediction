# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 15:05:24 2025

@author: Irshad Alam
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.getcwd())




# Use full path
model_path = 'D:/Machine Learning/House Pricing Prediction/savemodel/model.pkl'
scaler_path = 'D:/Machine Learning/House Pricing Prediction/savemodel/scaler.pkl'

df = pd.read_csv('D:/Machine Learning/House Pricing Prediction/dataset/Housing.csv')


# Load model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load scaler
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# Encode binary columns (to match training)
binary_map = {'yes': 1, 'no': 0}
for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
    df[col] = df[col].map(binary_map)

df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Streamlit UI
st.title("🏠 House Price Prediction App")
st.markdown("Enter the house details below to get an estimated price.")

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

# Prepare the input in correct order
input_data = [
    area, bedrooms, bathrooms, stories, parking,
    1 if mainroad == "yes" else 0,
    1 if guestroom == "yes" else 0,
    1 if basement == "yes" else 0,
    1 if hotwater == "yes" else 0,
    1 if ac == "yes" else 0,
    1 if prefarea == "yes" else 0,
    1 if furnishing == "semi-furnished" else 0  # Only one dummy
]

# Prediction button
if st.button("Predict Price"):
    input_array = np.array([input_data])
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    st.success(f"🏡 Estimated House Price: ₹ {prediction[0]:,.2f}")

# Optional: Visualize price distribution
if st.checkbox("Show price distribution in dataset"):
    fig, ax = plt.subplots()
    ax.hist(df['price'], bins=30, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of House Prices")
    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Number of Houses")
    st.pyplot(fig)