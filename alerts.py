"""
alerts.py — Farmer Advice & Telegram Alerting Layer
Predictive Grain Post-Harvest Protection System
"""

import requests
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"    # Replace with your BotFather token
TELEGRAM_CHAT_ID   = "YOUR_CHAT_ID"      # Replace with your chat/group ID

# ── Advice Database ───────────────────────────────────────────────────────────
ADVICE_MATRIX = {
    "Normal": {
        "Maize":   "✅ Maize storage is SAFE. Maintain current ventilation. Check again in 48h.",
        "Sorghum": "✅ Sorghum storage is SAFE. Conditions are optimal. No action needed.",
        "Wheat":   "✅ Wheat storage is SAFE. Keep monitoring temperature weekly.",
        "default": "✅ Grain storage is SAFE. All parameters within normal range.",
    },
    "Aflatoxin_Mold": {
        "Maize":   (
            "🔴 CRITICAL — Aflatoxin/Mold Risk in MAIZE!\n"
            "🌡️ High humidity detected.\n"
            "📋 ACTIONS:\n"
            "  1. Separate wet grain from dry immediately.\n"
            "  2. Spread grain in sun for 4–6 hours.\n"
            "  3. Improve silo ventilation or open vents.\n"
            "  4. Test a sample with Aflatest strips.\n"
            "  5. Do NOT sell or consume suspect grain."
        ),
        "Sorghum": (
            "🔴 CRITICAL — Aflatoxin/Mold Risk in SORGHUM!\n"
            "🌡️ Humidity dangerously high.\n"
            "📋 ACTIONS:\n"
            "  1. Dry sorghum to below 13% moisture urgently.\n"
            "  2. Use hermetic bags after drying.\n"
            "  3. Remove visibly discolored grains.\n"
            "  4. Contact local extension officer if >20% affected."
        ),
        "Wheat":   (
            "🔴 CRITICAL — Mold Risk in WHEAT!\n"
            "🌡️ Temperature & humidity conditions favour mycotoxins.\n"
            "📋 ACTIONS:\n"
            "  1. Aerate the store immediately.\n"
            "  2. Turn grain to expose damp layers.\n"
            "  3. Reduce store temperature if possible.\n"
            "  4. Inspect for clumping — discard clumped grain."
        ),
        "default": (
            "🔴 CRITICAL — Aflatoxin/Mold Risk Detected!\n"
            "📋 ACTIONS: Dry grain immediately, improve ventilation, separate wet batches."
        ),
    },
    "Insect_Parasite": {
        "Maize":   (
            "🟠 WARNING — Weevil/Insect Infestation in MAIZE!\n"
            "📈 Rapid CO₂ spike — insects are active.\n"
            "📋 ACTIONS:\n"
            "  1. Inspect grain bags for live insects.\n"
            "  2. Apply Actellic Super dust (approved dosage).\n"
            "  3. Use hermetic PICS bags to suffocate insects.\n"
            "  4. Freeze small samples 4 days to kill larvae.\n"
            "  5. Clean and seal storage facility."
        ),
        "Sorghum": (
            "🟠 WARNING — Insect Activity in SORGHUM!\n"
            "📈 CO₂ levels indicate active pest respiration.\n"
            "📋 ACTIONS:\n"
            "  1. Sift grain through 2mm mesh to remove insects.\n"
            "  2. Apply diatomaceous earth (natural, food-safe).\n"
            "  3. Seal all cracks in storage structure.\n"
            "  4. Monitor daily for 1 week."
        ),
        "Wheat":   (
            "🟠 WARNING — Weevil/Insect Infestation in WHEAT!\n"
            "📈 CO₂ spike pattern matches Sitophilus granarius.\n"
            "📋 ACTIONS:\n"
            "  1. Urgently inspect for adult weevils and larvae.\n"
            "  2. Fumigate with approved phosphine tablets (trained user only).\n"
            "  3. Move grain to triple-layered hermetic bags.\n"
            "  4. Report to cooperative if infestation is >10 bags."
        ),
        "default": (
            "🟠 WARNING — Insect Infestation Detected!\n"
            "📋 ACTIONS: Inspect immediately, apply approved insecticide, use hermetic storage."
        ),
    },
}


def get_farmer_advice(scenario: str, grain_type: str) -> str:
    """Return tailored, actionable advice for the farmer."""
    grain_advice = ADVICE_MATRIX.get(scenario, {})
    return grain_advice.get(grain_type, grain_advice.get("default", "⚠️ Unknown risk. Contact your extension officer."))


def format_alert_message(prediction: dict) -> str:
    """Format a complete alert message from a prediction result."""
    now = datetime.now().strftime("%d %b %Y, %H:%M")
    inputs = prediction.get("inputs", {})

    advice = get_farmer_advice(prediction["scenario"], inputs.get("grain_type", "Unknown"))

    message = (
        f"🌾 *GRAIN MONITOR ALERT*\n"
        f"📅 {now}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🌽 Grain   : {inputs.get('grain_type', 'N/A')}\n"
        f"🌡️ Temp    : {inputs.get('temperature_c', 'N/A')}°C\n"
        f"💧 Humidity: {inputs.get('humidity_pct', 'N/A')}%\n"
        f"🌬️ CO₂     : {inputs.get('co2_ppm', 'N/A')} ppm\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🚦 Status  : *{prediction.get('risk_level', 'Unknown')}*\n"
        f"⚠️ Threat  : {prediction.get('threat_type', 'N/A')}\n"
        f"🎯 Confidence: {prediction.get('confidence', 0)}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"{advice}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Powered by GrainGuard AI_"
    )
    return message


def send_telegram_alert(
    prediction: dict,
    bot_token: str = TELEGRAM_BOT_TOKEN,
    chat_id: str = TELEGRAM_CHAT_ID,
    only_on_risk: bool = True,
) -> dict:
    """
    Send a Telegram alert message for a prediction result.

    Args:
        prediction   : Output from predict_grain_risk()
        bot_token    : Telegram bot token
        chat_id      : Target chat or group ID
        only_on_risk : If True, skip sending for 'Normal'/'Safe' predictions

    Returns:
        dict with keys: sent (bool), status_code, message_preview
    """
    if only_on_risk and prediction.get("risk_level") == "Safe":
        return {"sent": False, "reason": "Skipped — Safe prediction, no alert needed."}

    if bot_token == "YOUR_BOT_TOKEN":
        message = format_alert_message(prediction)
        print("\n📱 [TELEGRAM SIMULATION — No token set]\n")
        print(message)
        return {"sent": False, "reason": "Placeholder token — configure YOUR_BOT_TOKEN.", "preview": message}

    message = format_alert_message(prediction)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    try:
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        response.raise_for_status()
        return {
            "sent": True,
            "status_code": response.status_code,
            "message_preview": message[:120] + "...",
        }
    except requests.RequestException as e:
        return {"sent": False, "error": str(e), "message_preview": message[:120] + "..."}


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_prediction = {
        "scenario": "Insect_Parasite",
        "risk_level": "Warning",
        "threat_type": "Weevil / Insect Infestation Detected",
        "color": "orange",
        "confidence": 94.3,
        "inputs": {
            "grain_type": "Maize",
            "temperature_c": 28.5,
            "humidity_pct": 61.0,
            "co2_ppm": 2150.0,
        },
    }
    result = send_telegram_alert(sample_prediction)
    print("\nAlert result:", result)
