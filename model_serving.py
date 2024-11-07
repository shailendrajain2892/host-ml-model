from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle  # or use pickle or the appropriate library for your model type

# Define your FastAPI application
app = FastAPI()

# Declare a global variable to hold the model
model = None

# Define input data model using Pydantic
class PredictRequest(BaseModel):
    feature1: float
    feature2: float
    # add other features as needed

# Load the model at startup
@app.on_event("startup")
def load_model():
    global model
    try:
        # Save the trained model to disk
        model_filename = "logistic_regression_iris.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading the model: {e}")
        raise e

# Prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Prepare the input data for the model
    input_data = [[request.feature1, request.feature2]]  # Adjust as per your model's input format

    # Perform prediction
    try:
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")