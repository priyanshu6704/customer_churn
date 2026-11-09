
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    credit_score: int = Field(..., ge=300, le=900)
    country: str
    gender: str
    age: int = Field(..., ge=18, le=100)
    tenure: int = Field(..., ge=0, le=10)
    balance: float = Field(..., ge=0)
    products_number: int = Field(..., ge=1, le=4)
    credit_card: int = Field(..., ge=0, le=1)
    active_member: int = Field(..., ge=0, le=1)
    estimated_salary: float = Field(..., ge=0)


class ModelPrediction(BaseModel):
    prediction: int
    churn_probability: float
    result: str


class PredictionResponse(BaseModel):
    logistic_regression: ModelPrediction
    random_forest: ModelPrediction
    xgboost: ModelPrediction
