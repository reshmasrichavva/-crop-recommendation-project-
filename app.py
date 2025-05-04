from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import boto3
import os

# Initialize the FastAPI app
app = FastAPI()

# Initialize an S3 client using boto3
s3 = boto3.client('s3')

# Function to download the model from S3
def download_model_from_s3(bucket_name, model_key, local_filename):
    s3.download_file(bucket_name, model_key, local_filename)

# Define the filename for your model
model_filename = "crop_recommendation_model.pkl"

# Download the model file if it doesn't already exist
if not os.path.exists(model_filename):
    download_model_from_s3("your-bucket-name", "models/crop_recommendation_model.pkl", model_filename)

# Load the model after downloading
model = joblib.load(model_filename)

# Define the Pydantic model for input data
class CropInput(BaseModel):
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Create a route to get crop recommendations
@app.post("/predict/")
def predict(input_data: CropInput):
    # Extract input data
    temp = input_data.temperature
    humidity = input_data.humidity
    ph = input_data.ph
    rainfall = input_data.rainfall

    # Predict the crop using the model (replace this with your prediction logic)
    prediction = model.predict([[temp, humidity, ph, rainfall]])

    # Return the prediction
    return {"predicted_crop": prediction[0]}


