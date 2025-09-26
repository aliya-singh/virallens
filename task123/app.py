import uvicorn
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# -----------------------------
# Load pre-trained model & scaler
# -----------------------------
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI()

class LaptopFeatures(BaseModel):
    features: list  # Length must match training features

@app.post("/predict")
def predict(data: LaptopFeatures):
    input_data = [data.features]  # list of features
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    return {"predicted_price": round(float(prediction), 2)}

# Run server: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
