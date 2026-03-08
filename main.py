"""
main.py — FastAPI Backend
Predictive Grain Post-Harvest Protection System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
from contextlib import asynccontextmanager

from brain import predict_grain_risk, generate_synthetic_data, train_model, CSV_PATH, MODEL_PATH, ENCODER_PATH
from alerts import send_telegram_alert, get_farmer_advice

# ── State ─────────────────────────────────────────────────────────────────────
app_state = {"model": None, "encoder": None, "sim_index": 0, "sim_df": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load or train model on startup."""
    if not os.path.exists(CSV_PATH):
        print("📊 Generating synthetic dataset...")
        generate_synthetic_data()

    if not os.path.exists(MODEL_PATH):
        print("🤖 Training model — first run...")
        train_model()

    app_state["model"]   = joblib.load(MODEL_PATH)
    app_state["encoder"] = joblib.load(ENCODER_PATH)
    app_state["sim_df"]  = pd.read_csv(CSV_PATH)
    print("✅ Model loaded. API ready.")
    yield
    print("👋 Shutting down.")


app = FastAPI(
    title="GrainGuard AI — Post-Harvest Protection API",
    description="ML-powered grain storage threat detection for Kenyan smallholder farmers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────
class SensorPayload(BaseModel):
    grain_type: str  = Field(..., example="Maize", description="Maize | Sorghum | Wheat")
    temperature_c: float = Field(..., ge=-10, le=60, example=28.5)
    humidity_pct: float  = Field(..., ge=0, le=100, example=72.0)
    co2_ppm: float       = Field(..., ge=300, le=5000, example=1800.0)
    send_alert: bool     = Field(False, description="Send Telegram alert on risk")


class PredictionResponse(BaseModel):
    scenario: str
    risk_level: str
    threat_type: str
    color: str
    confidence: float
    advice: str
    alert_sent: bool
    inputs: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "online", "service": "GrainGuard AI", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    model_ready = app_state["model"] is not None
    return {"model_loaded": model_ready, "dataset_rows": len(app_state["sim_df"]) if app_state["sim_df"] is not None else 0}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(payload: SensorPayload):
    """
    Run ML prediction on sensor readings.
    Optionally fires a Telegram alert for non-safe scenarios.
    """
    valid_grains = ["Maize", "Sorghum", "Wheat"]
    if payload.grain_type not in valid_grains:
        raise HTTPException(status_code=422, detail=f"grain_type must be one of {valid_grains}")

    prediction = predict_grain_risk(
        grain_type=payload.grain_type,
        temp=payload.temperature_c,
        humidity=payload.humidity_pct,
        co2=payload.co2_ppm,
        model=app_state["model"],
        encoder=app_state["encoder"],
    )

    advice = get_farmer_advice(prediction["scenario"], payload.grain_type)
    alert_sent = False

    if payload.send_alert:
        alert_result = send_telegram_alert(prediction)
        alert_sent = alert_result.get("sent", False)

    return PredictionResponse(
        scenario=prediction["scenario"],
        risk_level=prediction["risk_level"],
        threat_type=prediction["threat_type"],
        color=prediction["color"],
        confidence=prediction["confidence"],
        advice=advice,
        alert_sent=alert_sent,
        inputs=prediction["inputs"],
    )


@app.get("/simulate-stream", tags=["Simulation"])
def simulate_stream():
    """
    Returns the next row from grain_data.csv and runs prediction on it.
    Simulates a real-time IoT sensor feed. Loops back to row 0 at end.
    """
    df = app_state["sim_df"]
    idx = app_state["sim_index"] % len(df)
    row = df.iloc[idx]

    prediction = predict_grain_risk(
        grain_type=str(row["grain_type"]),
        temp=float(row["temperature_c"]),
        humidity=float(row["humidity_pct"]),
        co2=float(row["co2_ppm"]),
        model=app_state["model"],
        encoder=app_state["encoder"],
    )

    advice = get_farmer_advice(prediction["scenario"], str(row["grain_type"]))
    app_state["sim_index"] += 1

    return {
        "row_index": int(idx),
        "total_rows": len(df),
        "ground_truth_scenario": str(row["scenario"]),
        "prediction": prediction,
        "advice": advice,
    }


@app.post("/reset-simulation", tags=["Simulation"])
def reset_simulation():
    """Reset the simulation stream back to row 0."""
    app_state["sim_index"] = 0
    return {"message": "Simulation reset to row 0."}


@app.get("/dataset-stats", tags=["Data"])
def dataset_stats():
    """Return summary statistics of the training dataset."""
    df = app_state["sim_df"]
    return {
        "total_rows": len(df),
        "scenario_distribution": df["scenario"].value_counts().to_dict(),
        "grain_distribution": df["grain_type"].value_counts().to_dict(),
        "co2_stats": df["co2_ppm"].describe().round(2).to_dict(),
        "humidity_stats": df["humidity_pct"].describe().round(2).to_dict(),
        "temperature_stats": df["temperature_c"].describe().round(2).to_dict(),
    }


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
