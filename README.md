# 🌾 GrainGuard AI — Predictive Grain Post-Harvest Protection System

Detect invisible storage threats (Mold, Aflatoxin, Weevils) before they become visible.
Built for Kenyan smallholder farmers. Targets 10–30% reduction in post-harvest loss.

---

## 📁 Project Structure

```
grain_monitor/
├── brain.py          ← ML layer: data generation, model training, prediction
├── main.py           ← FastAPI backend (REST API)
├── alerts.py         ← Telegram alerting + farmer advice engine
├── dashboard.py      ← Streamlit dashboard UI
├── requirements.txt  ← All dependencies
└── README.md         ← This file
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate data + train model (one-time)
```bash
python brain.py
```
This creates `grain_data.csv` (1,000 rows) and `grain_model.joblib`.
Expect ~98%+ accuracy on the synthetic dataset.

### 3. Launch the Dashboard (recommended entry point)
```bash
streamlit run dashboard.py
```
The dashboard auto-trains the model on first launch if needed.  
When using **Live Simulation** you can pick a grain type in the sidebar; the
stream will then iterate through the existing rows for that grain in a fixed
order (no random swapping between types).

### 4. Launch the API (optional, for integrations)
```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

---

## 📡 API Endpoints

| Method | Endpoint            | Description                              |
|--------|---------------------|------------------------------------------|
| GET    | `/`                 | Health check                             |
| GET    | `/health`           | Model status                             |
| POST   | `/predict`          | Run ML on sensor payload                 |
| GET    | `/simulate-stream`  | Next row from CSV (simulates IoT sensor) |
| POST   | `/reset-simulation` | Reset stream to row 0                    |
| GET    | `/dataset-stats`    | Training data summary                    |

### Example `/predict` payload
```json
{
  "grain_type": "Maize",
  "temperature_c": 30.5,
  "humidity_pct": 82.0,
  "co2_ppm": 1350.0,
  "send_alert": false
}
```

---

## 📱 Telegram Alerts Setup

1. Open Telegram → search `@BotFather` → `/newbot` → copy your token
2. Send a message to your bot, then visit:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
   Copy the `chat.id` value.
3. **(New)** You no longer need to edit `alerts.py` directly.
   Instead, launch the **dashboard**, open the sidebar and make sure the
   **Send alerts to phone** toggle is on. Two new fields will appear:
   * **Bot token** – paste the value from BotFather.
   * **Chat ID** – numeric ID from the `/getUpdates` response.
   These are stored in Streamlit session state and used by the running app.
4. When the toggle is enabled and both credentials are filled, any non‑safe
   prediction will trigger a message to your Telegram account (or group).

   If you prefer to configure statically, you can still set
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `alerts.py` as before,
   but the dashboard inputs offer a quicker workflow.

---

## 🧠 ML Model Details

| Parameter       | Value                   |
|-----------------|-------------------------|
| Algorithm       | Random Forest Classifier |
| Trees           | 100                     |
| Features        | grain_type, temp, humidity, CO₂ |
| Classes         | Normal / Aflatoxin_Mold / Insect_Parasite |
| Train/Test split| 80/20 stratified        |
| Expected accuracy | >97%                  |

### Threat Detection Logic (Research-backed thresholds)
| Scenario         | Humidity    | CO₂ (ppm)   | Temp     |
|------------------|-------------|-------------|----------|
| Normal (Safe)    | 35–65%      | 400–900     | 15–30°C  |
| Aflatoxin/Mold   | **>75%**    | 900–1,500   | 25–38°C  |
| Insect/Parasite  | 50–72%      | **>1,500**  | 22–35°C  |

---

## 🌍 Impact

- **Target users**: Kenyan maize, sorghum, wheat farmers
- **Problem**: 10–30% post-harvest loss from undetected mold & pests
- **Solution**: Low-cost IoT sensors + ML → early warning → actionable SMS/Telegram advice
- **Hardware**: Optimized for HP EliteBook 8GB RAM (no GPU required)
