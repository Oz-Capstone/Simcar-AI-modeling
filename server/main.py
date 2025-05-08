from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import os
import uvicorn

app = FastAPI()

# 모델 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "car_fraud_model.joblib"
SCALER_PATH = BASE_DIR / "model" / "car_fraud_scaler.joblib"


# 모델과 스케일러 로드
def load_model_and_scaler():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다. 경로: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"스케일러 파일을 찾을 수 없습니다. 경로: {SCALER_PATH}"
        )

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        raise Exception(f"모델 로딩 중 오류 발생: {str(e)}")


try:
    model, scaler = load_model_and_scaler()
    print(f"모델 로딩 성공!\n모델 경로: {MODEL_PATH}\n스케일러 경로: {SCALER_PATH}")
except Exception as e:
    print(f"오류 발생: {str(e)}")
    raise


class CarInfo(BaseModel):
    brand: int
    model: int
    price: float
    productionYear: int
    mileage: float
    has_image: int
    insuranceHistory: int
    inspectionHistory: int
    region: int


@app.post("/predict")
async def predict_fraud(car_info: CarInfo):
    try:
        input_data = pd.DataFrame([car_info.dict()])

        numeric_features = [
            "price",
            "productionYear",
            "mileage",
            "insuranceHistory",
            "inspectionHistory",
        ]
        input_data[numeric_features] = scaler.transform(input_data[numeric_features])

        fraud_probability = float(model.predict_proba(input_data)[0][1])

        return {
            "success": True,
            "fraud_probability": fraud_probability,
            "probability_percentage": f"{fraud_probability:.2%}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def root():
    return {"message": "Car Fraud Detection API"}


if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # reload 기능 비활성화
    )
