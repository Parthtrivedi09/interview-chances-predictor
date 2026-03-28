import os

# ✅ Ensure ffmpeg is available (for Whisper)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

import joblib
import numpy as np
import pandas as pd
from voice_pipeline import get_soft_skill_score


# Load everything
data = joblib.load("log_reg.pkl") #load the logistic regression model

model = data["model"]
scaler = data["scaler"]
features = data["features"]

def predict(input_dict):
    import pandas as pd

    # Ensure correct order (CRITICAL FIX)
    input_df = pd.DataFrame([input_dict])[features]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict probability
    prob = model.predict_proba(input_scaled)[0][1]

    return prob


def generate_feedback(input_dict):
    strengths = []
    weaknesses = []

    # --- CGPA ---
    if input_dict["CGPA"] >= 8.5:
        strengths.append("Strong academic performance")
    elif input_dict["CGPA"] >= 7.5:
        strengths.append("Decent academic performance")
    else:
        weaknesses.append("Low CGPA, needs improvement")

    # --- Internships ---
    if input_dict["Internships"] >= 2:
        strengths.append("Good industry exposure through internships")
    else:
        weaknesses.append("Lack of internship experience")

    # --- Projects ---
    if input_dict["Projects"] >= 3:
        strengths.append("Good number of projects")
    else:
        weaknesses.append("Need more hands-on projects")

    # --- Aptitude ---
    if input_dict["AptitudeTestScore"] >= 80:
        strengths.append("Strong aptitude skills")
    elif input_dict["AptitudeTestScore"] >= 60:
        strengths.append("Average aptitude skills")
    else:
        weaknesses.append("Weak aptitude skills")

    # --- Soft Skills ---
    if input_dict["SoftSkillsRating"] >= 4.0:
        strengths.append("Good communication and soft skills")
    else:
        weaknesses.append("Needs improvement in communication skills")

    # --- Extracurricular (BINARY) ---
    if input_dict["ExtracurricularActivities"] == 1:
        strengths.append("Active in extracurricular activities")
    else:
        weaknesses.append("No extracurricular involvement")

    # --- Workshops ---
    if input_dict["Workshops/Certifications"] >= 2:
        strengths.append("Good certification profile")
    else:
        weaknesses.append("Few workshops/certifications")

    # --- Placement Training (BINARY) ---
    if input_dict["PlacementTraining"] == 1:
        strengths.append("Has undergone placement training")
    else:
        weaknesses.append("No placement training")

    return strengths, weaknesses


# ===============================
# 🎤 AUDIO INPUT (IMPROVED)
# ===============================

audio_path = r"C:\Users\Parth Trivedi\Desktop\CODING\MachineLearning\interview-chances-predictor\parth_audio_sample.m4a"

# ✅ Check file exists (prevents crash)
print("Audio exists:", os.path.exists(audio_path))
if not os.path.exists(audio_path):
    raise FileNotFoundError("Audio file not found. Check path.")

# ✅ Generate voice score safely (fallback added)
try:
    voice_score = get_soft_skill_score(audio_path)
    print(f"🎤 Voice Score (0–5): {round(voice_score,2)}")
except Exception as e:
    print("Voice processing failed:", e)
    voice_score = 2.5  # fallback neutral score


# ===============================
# 📊 MODEL INPUT
# ===============================

sample = {
    "CGPA": 9.4,
    "Internships": 0,
    "Projects": 2,
    "AptitudeTestScore": 85,
    "SoftSkillsRating": voice_score,
    "ExtracurricularActivities": 1,
    "Workshops/Certifications": 0,
    "PlacementTraining": 1
}

prob = predict(sample)
strengths, weaknesses = generate_feedback(sample)

print(f"\nChance: {round(prob*100,2)}%")
print("Strengths:", strengths)
print("Weaknesses:", weaknesses)


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