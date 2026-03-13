"""
brain.py — Data Generation & ML Layer
Predictive Grain Post-Harvest Protection System
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os

# ── Constants ────────────────────────────────────────────────────────────────
CSV_PATH = "grain_data.csv"
MODEL_PATH = "grain_model.joblib"
ENCODER_PATH = "label_encoders.joblib"

GRAIN_TYPES = ["Maize", "Sorghum", "Wheat"]
SCENARIOS = ["Normal", "Aflatoxin_Mold", "Insect_Parasite"]

RISK_MAP = {
    # keep both the newer descriptive keys as well as the old short forms
    # so existing data/files that might reference the older format still work.
    "Normal": {
        "risk_level": "Safe", "level": "Safe",
        "threat_type": "No Threat Detected", "threat": "No Threat Detected",
        "color": "green"
    },
    "Aflatoxin_Mold": {
        "risk_level": "Critical", "level": "Critical",
        "threat_type": "Aflatoxin / Mold Risk Detected", "threat": "Aflatoxin / Mold Risk Detected",
        "color": "red"
    },
    "Insect_Parasite": {
        "risk_level": "Warning", "level": "Warning",
        "threat_type": "Weevil / Insect Infestation Detected", "threat": "Weevil / Insect Infestation Detected",
        "color": "orange"
    },
}
# ── Synthetic Data Generation ─────────────────────────────────────────────────
def generate_synthetic_data(n_rows: int = 1000, save_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Generate research-backed synthetic sensor data for grain storage scenarios.
    Realistic ranges sourced from FAO post-harvest guidelines.
    """
    rng = np.random.default_rng(42)
    records = []

    rows_per_cell = n_rows // (len(GRAIN_TYPES) * len(SCENARIOS))
    remainder = n_rows % (len(GRAIN_TYPES) * len(SCENARIOS))

    for grain in GRAIN_TYPES:
        for scenario in SCENARIOS:
            n = rows_per_cell + (1 if remainder > 0 else 0)
            remainder = max(0, remainder - 1)

            if scenario == "Normal":
                # Safe storage: low humidity, stable temp, baseline CO2
                if grain == "Maize":
                    temp     = rng.uniform(18, 28, n)
                    humidity = rng.uniform(40, 65, n)
                    co2      = rng.uniform(400, 900, n)
                elif grain == "Sorghum":
                    temp     = rng.uniform(20, 30, n)
                    humidity = rng.uniform(35, 60, n)
                    co2      = rng.uniform(400, 850, n)
                else:  # Wheat
                    temp     = rng.uniform(15, 25, n)
                    humidity = rng.uniform(40, 60, n)
                    co2      = rng.uniform(400, 800, n)

            elif scenario == "Aflatoxin_Mold":
                # High humidity (>75%) is the KILLER signal for mold/aflatoxin
                temp     = rng.uniform(26, 40, n)
                humidity = rng.uniform(76, 98, n)        # Stricter: Always above 75%
                co2      = rng.uniform(950, 1600, n)     # Moderate metabolic rise

            else:  # Insect_Parasite
                # Massive CO2 spike is the KILLER signal for insects/weevils
                temp     = rng.uniform(24, 36, n)
                humidity = rng.uniform(45, 70, n)        # Keep humidity lower so it doesn't look like mold
                co2      = rng.uniform(1600, 4500, n)    # Stronger spike for clear detection

            for i in range(n):
                records.append({
                    "grain_type": grain,
                    "temperature_c": round(float(temp[i]), 2),
                    "humidity_pct": round(float(humidity[i]), 2),
                    "co2_ppm": round(float(co2[i]), 2),
                    "scenario": scenario,
                })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Generated {len(df)} rows → {save_path}")
    return df


# ── Model Training ────────────────────────────────────────────────────────────
def train_model(csv_path: str = CSV_PATH):
    """Train a Random Forest Classifier and persist it to disk."""
    df = pd.read_csv(csv_path)

    # Encode grain_type
    le_grain = LabelEncoder()
    df["grain_encoded"] = le_grain.fit_transform(df["grain_type"])

    features = ["grain_encoded", "temperature_c", "humidity_pct", "co2_ppm"]
    X = df[features]
    y = df["scenario"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n🌾 Model Training Complete")
    print(f"   Accuracy : {acc * 100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le_grain, ENCODER_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")
    print(f"✅ Encoder saved → {ENCODER_PATH}")
    return clf, le_grain, acc


# ── Prediction Function ───────────────────────────────────────────────────────
def predict_grain_risk(
    grain_type: str,
    temp: float,
    humidity: float,
    co2: float,
    model=None,
    encoder=None,
) -> dict:
    """
    Predict storage risk for given sensor readings.

    Returns:
        dict with keys: scenario, risk_level, threat_type, color, confidence
            (for backward compatibility the older keys ``level`` and ``threat``
            are also preserved).
    """
    if model is None or encoder is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Run train_model() first.")
        model   = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)

    grain_enc = encoder.transform([grain_type])[0]
    X = pd.DataFrame([[grain_enc, temp, humidity, co2]],
                     columns=["grain_encoded", "temperature_c", "humidity_pct", "co2_ppm"])

    scenario  = model.predict(X)[0]
    proba     = model.predict_proba(X)[0]
    confidence = float(max(proba)) * 100

    result = RISK_MAP[scenario].copy()
    # RISK_MAP entries originally used short keys ("level"/"threat") but
    # were later updated to explicit names.  Maintain backwards-compatibility
    # while avoiding accidentally overwriting the correct values.
    #
    # * If the map already has ``risk_level`` / ``threat_type`` we keep them.
    # * Otherwise try the old ``level``/``threat`` keys.
    if "risk_level" not in result or result.get("risk_level") is None:
        result["risk_level"] = result.get("level")
    if "threat_type" not in result or result.get("threat_type") is None:
        result["threat_type"] = result.get("threat")

    result["scenario"]   = scenario
    result["confidence"] = round(confidence, 1)
    result["inputs"]     = {
        "grain_type": grain_type,
        "temperature_c": temp,
        "humidity_pct": humidity,
        "co2_ppm": co2,
    }
    return result


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🌾 Grain Monitor — Brain Initialization")
    print("=" * 55)

    generate_synthetic_data()
    clf, le, acc = train_model()

    # Quick sanity checks
    print("\n🔬 Sample Predictions:")
    tests = [
        ("Maize",   22, 55, 700),    # Normal
        ("Sorghum", 30, 82, 1200),   # Aflatoxin/Mold
        ("Wheat",   28, 60, 2200),   # Insect
    ]
    for grain, t, h, c in tests:
        r = predict_grain_risk(grain, t, h, c, clf, le)
        level  = r.get("level",      r.get("risk_level", "?"))
        threat = r.get("threat",     r.get("threat_type", "?"))
        conf   = r.get("confidence", 0)
        print(f"  {grain:8} | Temp:{t}°C Hum:{h}% CO₂:{c}ppm"
              f"  →  [{level:8}] {threat} ({conf}%)")
