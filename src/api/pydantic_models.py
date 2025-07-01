from pydantic import BaseModel
from typing import List

class CustomerData(BaseModel):
    # Define all features as fields, e.g.:
    age: int
    income: float
    # Add all your model’s input features here with appropriate types

class PredictionResponse(BaseModel):
    risk_probability: float
