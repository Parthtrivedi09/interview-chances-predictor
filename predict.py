import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

import joblib
import numpy as np
import pandas as pd

# Load model
data = joblib.load("log_reg.pkl")
model = data["model"]
scaler = data["scaler"]
features = data["features"]

def predict(input_dict):
    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)

    prob = model.predict_proba(input_scaled)[0][1]

    print("Predicted probability:", prob)  # ✅ DEBUG

    return prob

def generate_feedback(input_dict):
    strengths = []
    weaknesses = []
    suggestions = []

    # CGPA
    if input_dict["CGPA"] >= 8.5:
        strengths.append("Strong academic performance")
    else:
        weaknesses.append("Low CGPA")
        suggestions.append("Focus on improving core subjects and maintaining consistency in academics")

    # Internships
    if input_dict["Internships"] >= 2:
        strengths.append("Good internship experience")
    else:
        weaknesses.append("Lack of internships")
        suggestions.append("Apply for internships on LinkedIn, Internshala, and build real-world experience")

    # Projects
    if input_dict["Projects"] >= 3:
        strengths.append("Good project experience")
    else:
        weaknesses.append("Insufficient projects")
        suggestions.append("Build 2-3 strong projects showcasing real-world problem solving")

    # Aptitude
    if input_dict["AptitudeTestScore"] >= 80:
        strengths.append("Strong aptitude skills")
    else:
        weaknesses.append("Weak aptitude")
        suggestions.append("Practice aptitude daily using platforms like IndiaBix or PrepInsta")

    # Soft skills
    if input_dict["SoftSkillsRating"] >= 4:
        strengths.append("Good communication skills")
    else:
        weaknesses.append("Poor communication skills")
        suggestions.append("Practice speaking daily, record yourself, and improve clarity & confidence")

    # Extracurricular
    if input_dict["ExtracurricularActivities"] == 1:
        strengths.append("Active in extracurricular activities")
    else:
        weaknesses.append("No extracurricular involvement")
        suggestions.append("Participate in clubs, events, or leadership roles")

    # Workshops
    if input_dict["Workshops/Certifications"] >= 2:
        strengths.append("Good certifications")
    else:
        weaknesses.append("Lack of certifications")
        suggestions.append("Complete certifications in ML, DSA, or relevant domains")

    # Placement training
    if input_dict["PlacementTraining"] == 1:
        strengths.append("Placement training completed")
    else:
        weaknesses.append("No placement training")
        suggestions.append("Join placement preparation programs or mock interview sessions")

    return strengths, weaknesses, suggestions


# LOGISTIC REGRESSION MODEL PARAMETERS IMPORTANCE

#                      Feature  Coefficient
# 0          AptitudeTestScore     0.709133
# 4  ExtracurricularActivities     0.465059
# 6          PlacementTraining     0.410735
# 1                   Projects     0.332311
# 2           SoftSkillsRating     0.313825
# 3                       CGPA     0.303405
# 5   Workshops/Certifications     0.143096
# 7                Internships     0.013033