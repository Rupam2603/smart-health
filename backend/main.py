from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
import requests
from typing import List, Optional, Dict, Any

app = FastAPI(title="Smart Health Assistant API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ML model
MODEL_PATH = os.path.join("models", "health_model.pkl")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

load_model()

# Data Models
class HealthData(BaseModel):
    glucose: float
    blood_pressure: float
    insulin: float
    bmi: float
    age: int

class SymptomData(BaseModel):
    text: str

# Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to Smart Health Assistant API"}

@app.post("/predict")
async def predict_health(data: HealthData):
    if model is None:
        load_model()
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    input_features = np.array([[data.glucose, data.blood_pressure, data.insulin, data.bmi, data.age]])
    prediction = int(model.predict(input_features)[0])
    
    # Detailed Insights Logic
    insights = []
    if data.glucose > 140: insights.append("High Glucose: Your blood sugar level is elevated. Consider reducing refined sugars.")
    if data.bmi > 30: insights.append("BMI Alert: A BMI over 30 indicates obesity, which is a major risk factor.")
    if data.blood_pressure > 90: insights.append("BP Monitoring: Your diastolic blood pressure is higher than the recommended range.")
    
    detailed_reports = {
        0: {
            "prediction": "Healthy",
            "summary": "Excellent! Your clinical parameters fall within the normal range.",
            "recommendations": [
                "Maintain a balanced diet rich in fiber and lean proteins.",
                "Continue regular physical activity (at least 150 mins/week).",
                "Schedule a routine checkup once a year."
            ],
            "risk_score": "Low"
        },
        1: {
            "prediction": "Pre-diabetes",
            "summary": "Caution: You are showing early signs of metabolic stress that could lead to diabetes.",
            "recommendations": [
                "Adopt a Low-Glycemic Index (GI) diet immediately.",
                "Incorporate 30 minutes of brisk walking daily.",
                "Monitor fasting blood sugar levels weekly."
            ],
            "risk_score": "Moderate"
        },
        2: {
            "prediction": "Diabetes Risk",
            "summary": "High Alert: Your current clinical profile suggests a significant risk of Type 2 Diabetes.",
            "recommendations": [
                "Consult an Endocrinologist for a diagnostic HbA1c test.",
                "Strict carbohydrate management and weight control are vital.",
                "Evaluate your medication needs with a healthcare provider."
            ],
            "risk_score": "High"
        }
    }
    
    report = detailed_reports[prediction]
    
    return {
        "status": "success",
        "prediction_code": prediction,
        "prediction": report["prediction"],
        "summary": report["summary"],
        "recommendations": report["recommendations"],
        "risk_score": report["risk_score"],
        "insights": insights if insights else ["All individual parameters are within manageable ranges."]
    }

@app.post("/symptoms")
async def check_symptoms(data: SymptomData):
    text = data.text.lower()
    
    # Simple NLP Triage Logic
    triage = {
        "fever": "You might have a common cold or infection. Rest and stay hydrated.",
        "cough": "Possible respiratory irritation. If persistent, check for allergies or infections.",
        "headache": "Commonly caused by stress or dehydration. Try resting in a dark room.",
        "pain": "General pain noted. If acute or severe, seek medical attention.",
        "default": "Your symptoms are noted. Please monitor closely and consult a healthcare provider if they worsen."
    }
    
    response = triage.get("default")
    for key in triage:
        if key in text:
            response = triage[key]
            break
            
    return {"analysis": response}

@app.get("/emergency-contacts")
async def get_emergency_contacts(country: str = "Global"):
    contacts = {
        "India": {
            "ambulance": "108", 
            "fire": "101", 
            "police": "100",
            "woman_child": "181"
        },
        "USA": {
            "ambulance": "911", 
            "fire": "911", 
            "police": "911",
            "woman_child": "911"
        },
        "UK": {
            "ambulance": "999", 
            "fire": "999", 
            "police": "999",
            "woman_child": "999"
        },
        "Global": {
            "ambulance": "112", 
            "fire": "112", 
            "police": "112",
            "woman_child": "112"
        }
    }
    return contacts.get(country, contacts["Global"])

@app.get("/doctors")
async def get_doctors(lat: float, lng: float):
    # Mock doctor discovery based on location
    return [
        {"name": "Dr. Sarah Johnson", "specialty": "Cardiologist", "distance": "0.8 km", "rating": 4.9},
        {"name": "Dr. Michael Chen", "specialty": "General Physician", "distance": "1.5 km", "rating": 4.7},
        {"name": "Dr. Amit Sharma", "specialty": "Endocrinologist", "distance": "2.1 km", "rating": 4.8},
        {"name": "Dr. Elena Rodriguez", "specialty": "Pediatrician", "distance": "3.4 km", "rating": 4.6}
    ]

@app.get("/hospitals")
async def get_hospitals(city: str, api_key: Optional[str] = None):
    # Mock data
    return [
        {"name": "City General Hospital", "distance": "2.5 km", "time": "10 mins"},
        {"name": "St. Luke's Medical Center", "distance": "4.1 km", "time": "15 mins"},
        {"name": "Central Health Clinic", "distance": "1.2 km", "time": "5 mins"},
        {"name": "Emergency Care Unit", "distance": "3.8 km", "time": "12 mins"},
        {"name": "Wellness Hospital", "distance": "5.0 km", "time": "20 mins"}
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
