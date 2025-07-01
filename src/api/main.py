from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from src.api.pydantic_models import CustomerData, PredictionResponse

app = FastAPI(title="Credit Scoring API")

# Load best model from MLflow Registry
MODEL_NAME = "credit_scoring_best_model"
STAGE = "Production"  # or "Staging" based on your model registry

print("Loading model from MLflow registry...")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{STAGE}")

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        input_df = customer.dict()
        # Wrap in list/dict to create 1-row DataFrame
        import pandas as pd
        input_df = pd.DataFrame([input_df])

        # Predict risk probability (assume binary classification, prob of positive class)
        risk_proba = model.predict_proba(input_df)[:, 1][0]
        return PredictionResponse(risk_probability=risk_proba)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
