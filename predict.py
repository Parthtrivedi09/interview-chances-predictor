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

    if input_dict["CGPA"] >= 8.5:
        strengths.append("Strong academic performance")
    else:
        weaknesses.append("Low CGPA")

    if input_dict["Internships"] >= 2:
        strengths.append("Good internships")
    else:
        weaknesses.append("Lack of internships")

    if input_dict["Projects"] >= 3:
        strengths.append("Good projects")
    else:
        weaknesses.append("Need more projects")

    if input_dict["AptitudeTestScore"] >= 80:
        strengths.append("Strong aptitude")
    else:
        weaknesses.append("Weak aptitude")

    if input_dict["SoftSkillsRating"] >= 4:
        strengths.append("Good communication")
    else:
        weaknesses.append("Improve communication")

    if input_dict["ExtracurricularActivities"] == 1:
        strengths.append("Extracurricular active")
    else:
        weaknesses.append("No extracurricular")

    if input_dict["Workshops/Certifications"] >= 2:
        strengths.append("Good certifications")
    else:
        weaknesses.append("Few certifications")

    if input_dict["PlacementTraining"] == 1:
        strengths.append("Placement training done")
    else:
        weaknesses.append("No placement training")

    return strengths, weaknesses


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