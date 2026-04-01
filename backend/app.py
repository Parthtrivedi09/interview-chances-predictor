
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, UploadFile, File, Form
import shutil
import joblib
from voice_pipeline import get_soft_skill_score
from predict import predict, generate_feedback
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
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
    cgpa: float = Form(...),
    internships: int = Form(...),
    projects: int = Form(...),
    aptitude: float = Form(...),
    extracurricular: int = Form(...),
    workshops: int = Form(...),
    placement_training: int = Form(...)
):
    # 🔴 Check missing values
    if None in [cgpa, internships, projects, aptitude, extracurricular, workshops, placement_training]:
        return {"error": "All fields are required"}

    # Save audio
    audio_path = f"temp_{file.filename}"
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Voice score
    voice_score = get_soft_skill_score(audio_path)

    # Create sample
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

    print("Sample received:", sample)  # ✅ DEBUG

    # Prediction
    prob = predict(sample)
    strengths, weaknesses, suggestions = generate_feedback(sample)

    return {
    "probability": round(prob * 100, 2),
    "voice_score": voice_score,
    "strengths": strengths,
    "weaknesses": weaknesses,
    "suggestions": suggestions
    }




#to start server
# python -m uvicorn backend.app:app --reload
