import streamlit as st
import pandas as pd
import numpy as np
import joblib
from predictor import predict_score

model = joblib.load("models/linear_regression_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("Student Performance Prediction")

import streamlit as st

# Numerical Inputs
hours_studied = st.number_input("Hours Studied", min_value=0, max_value=24)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=24)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100)
tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0)
physical_activity = st.number_input("Physical Activity (Hours/Week)", min_value=0)

# Categorical Inputs
parental_involvement = st.selectbox(
    "Parental Involvement",
    ["Low", "Medium", "High"]
)

access_to_resources = st.selectbox(
    "Access to Resources",
    ["Low", "Medium", "High"]
)

extracurricular_activities = st.selectbox(
    "Extracurricular Activities",
    ["Yes", "No"]
)

motivation_level = st.selectbox(
    "Motivation Level",
    ["Low", "Medium", "High"]
)

internet_access = st.selectbox(
    "Internet Access",
    ["Yes", "No"]
)

family_income = st.selectbox(
    "Family Income",
    ["Low", "Medium", "High"]
)

teacher_quality = st.selectbox(
    "Teacher Quality",
    ["Low", "Medium", "High"]
)

school_type = st.selectbox(
    "School Type",
    ["Public", "Private"]
)

peer_influence = st.selectbox(
    "Peer Influence",
    ["Negative", "Neutral", "Positive"]
)

learning_disabilities = st.selectbox(
    "Learning Disabilities",
    ["Yes", "No"]
)

parental_education_level = st.selectbox(
    "Parental Education Level",
    ["High School", "College", "Postgraduate"]
)

distance_from_home = st.selectbox(
    "Distance from Home",
    ["Near", "Moderate", "Far"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

input_data = {
    "Hours_Studied": hours_studied,
    "Attendance": attendance,
    "Sleep_Hours": sleep_hours,
    "Previous_Scores": previous_scores,
    "Tutoring_Sessions": tutoring_sessions,
    "Physical_Activity": physical_activity,
    "Parental_Involvement": parental_involvement,
    "Access_to_Resources": access_to_resources,
    "Extracurricular_Activities": extracurricular_activities,
    "Motivation_Level": motivation_level,
    "Internet_Access": internet_access,
    "Family_Income": family_income,
    "Teacher_Quality": teacher_quality,
    "School_Type": school_type,
    "Peer_Influence": peer_influence,
    "Learning_Disabilities": learning_disabilities,
    "Parental_Education_Level": parental_education_level,
    "Distance_from_Home": distance_from_home,
    "Gender": gender
}

if st.button("Predict"):

    prediction = predict_score(input_data)

    st.success(f"Predicted Exam Score: {prediction:.2f}")