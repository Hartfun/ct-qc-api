from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from predict import predict

app = FastAPI(
    title="CT QC Anomaly Detection API",
    description="ML ensemble model for CT scanner quality control",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CTRecord(BaseModel):
    """CT QC Test Record Schema"""
    serial_no: str = Field(..., alias="serial No")
    date: str = Field(..., alias="Date")
    slice_1_5: float = Field(..., alias="Slice thickness 1.5")
    slice_5: float = Field(..., alias="Slice thickness 5")
    slice_10: float = Field(..., alias="Slice thickness 10")
    kv_80: float = Field(..., alias="KV accuracy 80")
    kv_110: float = Field(..., alias="KV accuracy 110")
    kv_130: float = Field(..., alias="KV accuracy 130")
    timer_0_8: float = Field(..., alias="Accuracy Timer 0.8")
    timer_1: float = Field(..., alias="Accuracy Timer 1")
    timer_1_5: float = Field(..., alias="Accuracy Timer 1.5")
    dose_head: float = Field(..., alias="Radiation Dose Test (Head) 21.50")
    dose_body: float = Field(..., alias="Radiation Dose Test (Body) 10.60")
    leak_front: float = Field(..., alias="Radiation Leakage Levels (Front)")
    leak_back: float = Field(..., alias="Radiation Leakage Levels (Back)")
    leak_left: float = Field(..., alias="Radiation Leakage Levels (Left)")
    leak_right: float = Field(..., alias="Radiation Leakage Levels (Right)")

    class Config:
        populate_by_name = True


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "✅ OK", "model": "ct_qc_production v1.0"}


@app.post("/predict")
def run_prediction(record: CTRecord):
    """
    Predict QC acceptance for a single CT test.
    Returns anomaly scores and risk assessment.
    """
    try:
        input_dict = record.model_dump(by_alias=True)
        result = predict(input_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
def run_batch(records: list[CTRecord]):
    """Batch predict for multiple CT tests"""
    results = []
    for r in records:
        try:
            results.append(predict(r.model_dump(by_alias=True)))
        except Exception as e:
            results.append({"error": str(e), "serial_no": r.serial_no})
    return results


@app.get("/")
def root():
    """API documentation"""
    return {
        "message": "CT QC Anomaly Detection API",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "batch_predict": "POST /predict/batch"
        }
    }
