from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient

# Define input schema
class IrisInput(BaseModel):
    features: list[float]  # Example: [5.1, 3.5, 1.4, 0.2]

# Always fetch latest Production model
def load_production_model(model_name: str):
    client = MlflowClient()
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        raise ValueError(f"No Production model found for {model_name}")
    model_uri = f"models:/{model_name}/Production"
    return mlflow.sklearn.load_model(model_uri)

# Load model dynamically
model_name = "IrisClassifier"
model = load_production_model(model_name)

# Create FastAPI app
app = FastAPI(title="Iris Classifier API")

@app.post("/predict")
def predict(input_data: IrisInput):
    prediction = model.predict([input_data.features])
    return {"prediction": int(prediction[0])}
