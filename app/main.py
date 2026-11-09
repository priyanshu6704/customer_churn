
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.schemas import CustomerInput, PredictionResponse
from app.predictor import predict_all

import json

app=FastAPI(
    title="CustomerChurn",
    description="Customer churn prediction",
    version="1.0.0"
)

BASE_DIR=Path(__file__).resolve().parent
STATIC_DIR=BASE_DIR/"static"

METRICS_PATH=BASE_DIR.parent/"reports"/"model_metrics.json"

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)

@app.get("/health")
def health_check():
    return{
        "status":"healthy",
        "application":"Customer Churn"
    }

@app.get("/")
def home():
    return FileResponse("app/static/index.html")

@app.post("/predict",response_model=PredictionResponse)
def predict_churn(customer:CustomerInput):
    try:
        customer_data=customer.model_dump()
        predictions=predict_all(customer_data)
        return predictions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed:{str(e)}"
        )


@app.get("/models/metrics")
def get_model_metrics():
    try:
        if not METRICS_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail="Model metrics file not found"
            )

        with open(METRICS_PATH,"r") as file:
            metrics=json.load(file)

        return metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to load metrics: {str(e)}"
        )


