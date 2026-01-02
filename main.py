from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

origins_env = os.getenv("ALLOWED_ORIGINS")

ALLOWED_ORIGINS = (
    origins_env.split(",")
    if origins_env
    else ["*"]  # OK for MVP, restrict later
)

# -----------------------------
# Validate required env vars
# -----------------------------
if not MODEL_PATH:
    raise RuntimeError("MODEL_PATH is not set")

if not PROVINCE_ENCODER_PATH:
    raise RuntimeError("PROVINCE_ENCODER is not set")

if not EDUCATION_ENCODER_PATH:
    raise RuntimeError("EDUCATION_ENCODER is not set")

if not JOB_ENCODER_PATH:
    raise RuntimeError("JOB_ENCODER is not set")

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(
    title="Salary Prediction API",
    description="Predict salary based on user profile",
    version="1.0.0"
)

# -----------------------------
# Enable CORS
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
    provinsi: str
    pengalaman: int = Field(ge=0)
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
    try:
        domisili_encoded = province_encoder.transform([data.provinsi])[0]
        pendidikan_encoded = education_encoder.transform([data.pendidikan])[0]
        pekerjaan_encoded = job_encoder.transform([data.pekerjaan])[0]

        X = np.array([[
            domisili_encoded,
            data.pengalaman,
            pendidikan_encoded,
            pekerjaan_encoded
        ]])

        prediction = model.predict(X, verbose=0)
        salary = int(prediction[0][0])

        return {
            "nama": data.nama,
            "predicted_salary": salary
        }

    except Exception as e:
        # For MVP, simple error
        return {
            "nama": data.nama,
            "predicted_salary": 0
        }

# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}
