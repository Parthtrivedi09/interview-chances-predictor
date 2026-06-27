import sys                                      # Used to interact with Python runtime (path management, arguments, etc.)
import os                                       # Used for operating system related operations (paths, files, directories)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Adds the parent directory to Python's module search path so that files outside
# the backend folder (like voice_pipeline.py and predict.py) can be imported.

from fastapi import FastAPI, UploadFile, File, Form
# FastAPI -> Creates the API server
# UploadFile -> Handles uploaded files efficiently
# File -> Specifies that an input is a file
# Form -> Specifies that an input comes from form-data

import shutil
# Used to copy the uploaded audio file onto the disk.

import joblib
# Used to load the trained machine learning model (.pkl file).

from voice_pipeline import get_soft_skill_score
# Imports the function that analyzes the uploaded voice and returns a soft skill score.

from predict import predict, generate_feedback
# Imports two functions:
# predict() -> Returns placement probability
# generate_feedback() -> Generates strengths, weaknesses and suggestions.

from fastapi.middleware.cors import CORSMiddleware
# Middleware that allows frontend applications (React, Angular, etc.) to access this backend.

app = FastAPI()
# Creates the FastAPI application object.

# ------------------------ CORS ------------------------

app.add_middleware(
    CORSMiddleware,                 # Adds Cross-Origin Resource Sharing middleware.
    allow_origins=["*"],            # Allows requests from any origin.
    allow_credentials=True,         # Allows cookies/authentication credentials.
    allow_methods=["*"],            # Allows every HTTP method (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],            # Allows every request header.
)

# --------------------- Load Model ----------------------

data = joblib.load("log_reg.pkl")
# Loads the saved machine learning model from disk.

model = data["model"]
# Extracts the trained Logistic Regression model.

scaler = data["scaler"]
# Extracts the StandardScaler used during training.

features = data["features"]
# Extracts the list of feature names used during training.

# --------------------- Home API ------------------------

@app.get("/")
# Creates a GET endpoint at "/"

def home():
    # Function executed whenever someone visits the root URL.

    return {"message": "Interview Predictor API Running 🚀"}
    # Returns a simple JSON response indicating the API is working.

# ---------------- Prediction API -----------------------

@app.post("/predict")
# Creates a POST endpoint at "/predict"

async def predict_interview(
    file: UploadFile = File(...),             # Uploaded audio file (mandatory)
    cgpa: float = Form(...),                  # CGPA from form-data
    internships: int = Form(...),             # Number of internships
    projects: int = Form(...),                # Number of projects
    aptitude: float = Form(...),              # Aptitude score
    extracurricular: int = Form(...),         # Extracurricular activity score
    workshops: int = Form(...),               # Number of workshops/certifications
    placement_training: int = Form(...)       # Placement training completed (0/1)
):

    # ---------------- Missing Value Check ----------------

    if None in [cgpa, internships, projects, aptitude, extracurricular, workshops, placement_training]:
        # Checks if any required field is missing.

        return {"error": "All fields are required"}
        # Returns an error message if any field is missing.

    # ---------------- Save Uploaded Audio ----------------

    audio_path = f"temp_{file.filename}"
    # Creates a temporary filename using the uploaded filename.

    with open(audio_path, "wb") as buffer:
        # Opens a new file in write-binary mode.

        shutil.copyfileobj(file.file, buffer)
        # Copies the uploaded audio file into the newly created file.

    # ---------------- Voice Analysis ----------------

    voice_score = get_soft_skill_score(audio_path)
    # Passes the saved audio file to the voice analysis pipeline.
    # Returns a predicted soft skill score.

    # ---------------- Create Sample ----------------

    sample = {
        "CGPA": cgpa,                                    # Student CGPA
        "Internships": internships,                      # Number of internships
        "Projects": projects,                            # Number of projects
        "AptitudeTestScore": aptitude,                   # Aptitude score
        "SoftSkillsRating": voice_score,                 # Voice-based soft skill score
        "ExtracurricularActivities": extracurricular,    # Extracurricular score
        "Workshops/Certifications": workshops,           # Workshops attended
        "PlacementTraining": placement_training          # Placement training status
    }

    print("Sample received:", sample)
    # Prints the final feature dictionary in the terminal for debugging.

    # ---------------- Prediction ----------------

    prob = predict(sample)
    # Sends the sample to the ML model and gets placement probability.

    strengths, weaknesses, suggestions = generate_feedback(sample)
    # Generates personalized feedback based on the student's profile.

    # ---------------- Return Response ----------------

    return {
        "probability": round(prob * 100, 2),      # Converts probability into percentage
        "voice_score": voice_score,               # Returns calculated voice score
        "strengths": strengths,                   # List of strengths
        "weaknesses": weaknesses,                 # List of weaknesses
        "suggestions": suggestions                # List of improvement suggestions
    }

# ---------------- Run Server ----------------

# python -m uvicorn backend.app:app --reload
# Command to start the FastAPI development server.
# --reload automatically restarts the server whenever code changes.
