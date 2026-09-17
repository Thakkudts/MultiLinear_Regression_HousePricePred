import streamlit as st
import joblib

model = joblib.load("Multiple_Linear_Regression_House_Price.pkl")

st.title("House Price Prediction")

area = st.number_input("Enter Area (sq.ft):",min_value=600.0,max_value=3000.0,value=1500.0)

bedrooms = st.number_input("Enter Number of Bedrooms:",min_value=1,max_value=4,value=2)

floors = st.number_input("Enter Number of Floors:",min_value=0, max_value=10,value=1)

if st.button("Predict Price"):

    new_data = [[area, bedrooms, floors]]

    prediction = model.predict(new_data)

    price = prediction[0]

    st.success(f"Predicted House Price: {price:.2f} Lakhs")
