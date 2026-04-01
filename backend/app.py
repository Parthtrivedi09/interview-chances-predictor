import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File
import shutil
import joblib
import pandas as pd
from voice_pipeline import get_soft_skill_score
from predict import predict, generate_feedback



app = FastAPI()

# Load model once
data = joblib.load("log_reg.pkl")
model = data["model"]
scaler = data["scaler"]
features = data["features"]


@app.get("/")
def home():
    return {"message": "Interview Predictor API Running 🚀"}


@app.post("/predict")
async def predict_interview(
    file: UploadFile = File(...),
    cgpa: float = 0,
    internships: int = 0,
    projects: int = 0,
    aptitude: float = 0,
    extracurricular: int = 0,
    workshops: int = 0,
    placement_training: int = 0
):
    # Save uploaded audio
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🎤 Voice Score
    voice_score = get_soft_skill_score(audio_path)

    # Create input
    sample = {
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "AptitudeTestScore": aptitude,
        "SoftSkillsRating": voice_score,
        "ExtracurricularActivities": extracurricular,
        "Workshops/Certifications": workshops,
        "PlacementTraining": placement_training
    }

    # Prediction
    prob = predict(sample)
    strengths, weaknesses = generate_feedback(sample)

    return {
        "probability": round(prob * 100, 2),
        "voice_score": voice_score,
        "strengths": strengths,
        "weaknesses": weaknesses
    }



# python -m uvicorn backend.app:app --reload