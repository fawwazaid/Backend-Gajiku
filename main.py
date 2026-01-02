from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import joblib
import os

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

MODEL_PATH = os.getenv("MODEL_PATH")
PROVINCE_ENCODER_PATH = os.getenv("PROVINCE_ENCODER")
EDUCATION_ENCODER_PATH = os.getenv("EDUCATION_ENCODER")
JOB_ENCODER_PATH = os.getenv("JOB_ENCODER")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="Salary Prediction API",
    description="Predict take-home pay based on user profile",
    version="1.0.0"
)

# -----------------------------
# Enable CORS (frontend access)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ML model & encoders
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

province_encoder = joblib.load(PROVINCE_ENCODER_PATH)
education_encoder = joblib.load(EDUCATION_ENCODER_PATH)
job_encoder = joblib.load(JOB_ENCODER_PATH)

# -----------------------------
# Request schema
# -----------------------------
class SalaryRequest(BaseModel):
    nama: str
    domisili: str
    pengalaman: int
    pendidikan: str
    pekerjaan: str

# -----------------------------
# Response schema
# -----------------------------
class SalaryResponse(BaseModel):
    nama: str
    predicted_salary: int

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict", response_model=SalaryResponse)
def predict_salary(data: SalaryRequest):
    # Encode categorical variables
    domisili_encoded = province_encoder.transform([data.domisili])[0]
    pendidikan_encoded = education_encoder.transform([data.pendidikan])[0]
    pekerjaan_encoded = job_encoder.transform([data.pekerjaan])[0]

    # Prepare input for model
    X = np.array([[
        domisili_encoded,
        data.pengalaman,
        pendidikan_encoded,
        pekerjaan_encoded
    ]])

    # Predict salary
    prediction = model.predict(X)
    salary = int(prediction[0][0])

    return {
        "nama": data.nama,
        "predicted_salary": salary
    }

# -----------------------------
# Health check (optional)
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

