import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.pkl")

st.title("🚖 Taxi Demand Prediction App")
st.write("Enter inputs to predict taxi demand.")

zone = st.number_input("Zone ID", min_value=1, max_value=300, step=1)
hour = st.slider("Hour of Day", 0, 23)
day = st.slider("Day of Week (0=Mon , 6=Sun)", 0, 6)
month = st.slider("Month (1–12)", 1, 12)

if st.button("Predict Demand"):
    features = np.array([[zone, hour, day, month]])
    pred = model.predict(features)[0]
    st.success(f"Predicted Taxi Demand: {int(pred)} rides")
