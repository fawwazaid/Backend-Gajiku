import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data_salary.csv")

# -----------------------------
# Separate features & target
# -----------------------------
X = df[["provinsi", "pengalaman", "pendidikan", "pekerjaan"]]
y = df["gaji"]

# -----------------------------
# Create encoders
# -----------------------------
province_encoder = LabelEncoder()
education_encoder = LabelEncoder()
job_encoder = LabelEncoder()

X["provinsi"] = province_encoder.fit_transform(X["provinsi"])
X["pendidikan"] = education_encoder.fit_transform(X["pendidikan"])
X["pekerjaan"] = job_encoder.fit_transform(X["pekerjaan"])

# -----------------------------
# Save encoders
# -----------------------------
os.makedirs("encoders", exist_ok=True)
joblib.dump(province_encoder, "encoders/province_encoder.pkl")
joblib.dump(education_encoder, "encoders/education_encoder.pkl")
joblib.dump(job_encoder, "encoders/job_encoder.pkl")

# -----------------------------
# Train / test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42
)

# -----------------------------
# Build model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

# -----------------------------
# Train model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# -----------------------------
# Evaluate
# -----------------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:,.0f}")

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/salary_model.h5")

print("✅ Model and encoders saved successfully")

