from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

# Initialize the FastAPI app
app = FastAPI()

# Define the filename for your model
model_filename = "crop_recommendation_model.pkl"

# Check if model file exists locally
if not os.path.exists(model_filename):
    raise FileNotFoundError(f"{model_filename} not found. Please ensure the model file is in the project directory.")

# Load the model
model = joblib.load(model_filename)

# Define the Pydantic model for input data
class CropInput(BaseModel):
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Root route
@app.get("/")
def read_root():
    return {"message": "Crop Recommendation API is running"}

# Prediction route
@app.post("/predict/")
def predict(input_data: CropInput):
    features = [[
        input_data.temperature,
        input_data.humidity,
        input_data.ph,
        input_data.rainfall
    ]]
    prediction = model.predict(features)
    return {"predicted_crop": prediction[0]}


