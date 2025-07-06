# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

st.title("🏠 House Price Prediction App")
st.markdown("Upload the required files and enter house details below to get an estimated price.")

# --- Upload files ---
uploaded_model = st.file_uploader("🔁 Upload model.pkl", type="pkl")
uploaded_scaler = st.file_uploader("🧪 Upload scaler.pkl", type="pkl")
uploaded_csv = st.file_uploader("📄 Upload Housing.csv", type="csv")

if uploaded_model and uploaded_scaler and uploaded_csv:
    # Load model & scaler
    model = pickle.load(uploaded_model)
    scaler = pickle.load(uploaded_scaler)
    df = pd.read_csv(uploaded_csv)

    # Binary encoding
    binary_map = {'yes': 1, 'no': 0}
    for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
        df[col] = df[col].map(binary_map)

    # One-hot encode furnishingstatus
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

    st.success("✅ Files loaded successfully!")

    # --- Input fields ---
    st.subheader("🏗️ Enter House Features")

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

    # --- Prepare input data ---
    input_data = [
        area,
        bedrooms,
        bathrooms,
        stories,
        parking,
        1 if mainroad == "yes" else 0,
        1 if guestroom == "yes" else 0,
        1 if basement == "yes" else 0,
        1 if hotwater == "yes" else 0,
        1 if ac == "yes" else 0,
        1 if prefarea == "yes" else 0,
        1 if furnishing == "semi-furnished" else 0,
        1 if furnishing == "furnished" else 0
    ]

    # --- Prediction ---
    if st.button("🔍 Predict Price"):
        input_array = np.array([input_data])
        scaled_input = scaler.transform(input_array)
        predicted_price = model.predict(scaled_input)

        # Multiply by 1 lakh if model was trained on price in lakhs
        price_in_inr = predicted_price[0] * 1_00_000
        st.success(f"🏡 Estimated House Price: ₹ {price_in_inr:,.2f}")

    # --- Visualization ---
    if st.checkbox("📊 Show price distribution in dataset"):
        fig, ax = plt.subplots()
        ax.hist(df['price'] * 1_00_000, bins=30, color='skyblue', edgecolor='black')  # Assuming price was in lakhs
        ax.set_title("Distribution of House Prices")
        ax.set_xlabel("Price (₹)")
        ax.set_ylabel("Number of Houses")
        st.pyplot(fig)

else:
    st.warning("⚠️ Please upload all required files (`model.pkl`, `scaler.pkl`, and `Housing.csv`) to continue.")
