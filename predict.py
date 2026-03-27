
import joblib
import numpy as np

# Load everything
data = joblib.load("log_reg.pkl") #load the logistic regression model

model = data["model"]
scaler = data["scaler"]
features = data["features"]

def predict(input_dict):
    # Ensure correct order
    input_data = [input_dict[feature] for feature in features]

    input_array = np.array(input_data).reshape(1, -1)

    # Scale

    input_scaled = scaler.transform(input_array)

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
    if input_dict["SoftSkillsRating"] >= 4.2:
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


sample = {
    "CGPA": 9.4,
    "Internships": 0,
    "Projects": 2,
    "AptitudeTestScore": 85,
    "SoftSkillsRating": 4.3,
    "ExtracurricularActivities": 1,
    "Workshops/Certifications": 0,
    "PlacementTraining": 1
}

prob = predict(sample)
strengths, weaknesses = generate_feedback(sample)

print(f"Chance: {round(prob*100,2)}%")
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