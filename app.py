import streamlit as st
import pandas as pd
import joblib

#st.write("App started...")

# Load model
model = joblib.load("house_model.pkl")

st.title("🏠 Vidhu's Price Predictor")

# Inputs
area = st.number_input("Area", value=3000)
bedrooms = st.number_input("Bedrooms", value=3)
bathrooms = st.number_input("Bathrooms", value=2)
stories = st.number_input("Stories", value=2)

mainroad = st.selectbox("Main Road", [0, 1])
guestroom = st.selectbox("Guest Room", [0, 1])
basement = st.selectbox("Basement", [0, 1])
hotwaterheating = st.selectbox("Hot Water Heating", [0, 1])
airconditioning = st.selectbox("Air Conditioning", [0, 1])
parking = st.number_input("Parking", value=1)
prefarea = st.selectbox("Preferred Area", [0, 1])
furnishingstatus = st.selectbox("Furnishing", [0, 1, 2])

# Predict button
if st.button("Predict Price"):
    data = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])

    prediction = model.predict(data)

    st.success(f"Predicted Price: ₹ {int(prediction[0])}")