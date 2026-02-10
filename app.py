import streamlit as st
import numpy as np
from model import train_model

st.set_page_config(page_title="Student Performance Predictor")

st.title("🎓 Student Performance Predictor")
st.write("Predict final exam score using Machine Learning")

model = train_model()

hours = st.slider("Hours Studied per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
previous_score = st.slider("Previous Exam Score", 0, 100, 60)

if st.button("Predict"):
    data = np.array([[hours, attendance, previous_score]])
    prediction = model.predict(data)
    st.success(f"Predicted Final Score: {prediction[0]:.2f}")
