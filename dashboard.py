"""
dashboard.py — GrainGuard AI Dashboard (Streamlit)
Predictive Grain Post-Harvest Protection System
"""

import streamlit as st
import pandas as pd
import time
import os
import sys

# ── Page Config (MUST be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="GrainGuard AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap ML layer ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from brain import (
    generate_synthetic_data, train_model, predict_grain_risk,
    CSV_PATH, MODEL_PATH, ENCODER_PATH, RISK_MAP
)
from alerts import send_telegram_alert, get_farmer_advice, format_alert_message
import joblib

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;1,9..144,400&display=swap');

:root {
    --soil:    #2C1A0E;
    --grain:   #C8932A;
    --harvest: #E8B84B;
    --leaf:    #3D7A3B;
    --sky:     #E8EDE4;
    --danger:  #C0392B;
    --warn:    #E67E22;
    --safe:    #27AE60;
    --card-bg: #1A0F07;
    --panel:   #231408;
}

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: var(--soil) !important;
    color: var(--sky);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 2px solid var(--grain);
}

/* Header brand */
.brand-header {
    text-align: center;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid rgba(200,147,42,0.3);
    margin-bottom: 1.5rem;
}
.brand-title {
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--harvest);
    letter-spacing: -0.02em;
    line-height: 1;
}
.brand-sub {
    font-size: 0.65rem;
    color: rgba(232,237,228,0.5);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* Metric cards */
.metric-card {
    background: var(--card-bg);
    border: 1px solid rgba(200,147,42,0.25);
    border-radius: 6px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-card:hover { border-color: var(--grain); }
.metric-label {
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(232,237,228,0.45);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--harvest);
    line-height: 1;
}
.metric-unit {
    font-size: 0.7rem;
    color: rgba(232,237,228,0.4);
    margin-top: 0.2rem;
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 0.5rem 1.4rem;
    border-radius: 2px;
    font-weight: 700;
    letter-spacing: 0.1em;
    font-size: 0.85rem;
    text-transform: uppercase;
}
.status-safe     { background: rgba(39,174,96,0.15);  color: #2ECC71; border: 1px solid #27AE60; }
.status-warning  { background: rgba(230,126,34,0.15); color: #F39C12; border: 1px solid #E67E22; }
.status-critical { background: rgba(192,57,43,0.15);  color: #E74C3C; border: 1px solid #C0392B; }

/* Advice panel */
.advice-panel {
    background: var(--card-bg);
    border-left: 3px solid var(--grain);
    border-radius: 0 6px 6px 0;
    padding: 1.2rem 1.4rem;
    font-size: 0.82rem;
    line-height: 1.75;
    white-space: pre-wrap;
    font-family: 'Space Mono', monospace;
}

/* Confidence bar */
.conf-bar-wrap { background: rgba(255,255,255,0.07); border-radius: 2px; height: 6px; margin-top:0.6rem; }
.conf-bar      { height: 6px; border-radius: 2px; background: var(--harvest); transition: width 0.6s ease; }

/* Log table */
.log-row-safe     { color: #2ECC71; }
.log-row-warning  { color: #F39C12; }
.log-row-critical { color: #E74C3C; }

/* Divider */
.grain-divider {
    border: none;
    border-top: 1px solid rgba(200,147,42,0.2);
    margin: 1.5rem 0;
}

/* Streamlit overrides */
.stButton > button {
    background: var(--grain) !important;
    color: var(--soil) !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 3px !important;
    letter-spacing: 0.08em !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: var(--harvest) !important;
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label { color: var(--sky) !important; font-size: 0.72rem !important; letter-spacing: 0.1em !important; text-transform: uppercase; }

.stAlert { background: var(--card-bg) !important; border-radius: 4px !important; }

/* Pulse animation for active status */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.pulse { animation: pulse 2s infinite; }
</style>
""", unsafe_allow_html=True)


# ── Helper: ensure model is ready ─────────────────────────────────────────
@st.cache_resource(show_spinner="🌾 Training CeresGuard AI model…")
def load_system():
    if not os.path.exists(CSV_PATH):
        generate_synthetic_data()
    if not os.path.exists(MODEL_PATH):
        train_model()
    model   = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    df      = pd.read_csv(CSV_PATH)
    return model, encoder, df

model, encoder, sim_df = load_system()

# ── Session State ─────────────────────────────────────────────────────────
if "sim_index"    not in st.session_state: st.session_state.sim_index    = 0
if "sim_running"  not in st.session_state: st.session_state.sim_running  = False
if "last_pred"    not in st.session_state: st.session_state.last_pred    = None
if "history"      not in st.session_state: st.session_state.history      = []
if "manual_pred"  not in st.session_state: st.session_state.manual_pred  = None


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-header">
        <div class="brand-title">🌾 GrainGuard</div>
        <div class="brand-sub">Post-Harvest AI System</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Mode", ["📡 Live Simulation", "🔬 Manual Sensor Input"], label_visibility="collapsed")
    st.markdown('<hr class="grain-divider">', unsafe_allow_html=True)

    grain_type = st.selectbox(
        "Grain Type",
        ["Maize", "Sorghum", "Wheat"],
        help="Select the grain being monitored"
    )

    st.markdown('<hr class="grain-divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:rgba(232,237,228,0.4);margin-bottom:0.8rem;">Telegram Alerts</div>', unsafe_allow_html=True)
    send_alerts = st.toggle("Send alerts to phone", value=False)

    # when alerts are enabled, collect bot token & chat id from user
    bot_token = st.session_state.get("bot_token", "")
    chat_id   = st.session_state.get("chat_id", "")
    if send_alerts:
        bot_token = st.text_input("Bot token", bot_token, help="Generated by @BotFather")
        chat_id   = st.text_input("Chat ID", chat_id, help="Numeric ID from /getUpdates")
    # store back to session so it persists while app is open
    st.session_state.bot_token = bot_token
    st.session_state.chat_id   = chat_id

    st.markdown('<hr class="grain-divider">', unsafe_allow_html=True)

    # Dataset stats
    st.markdown('<div style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;color:rgba(232,237,228,0.4);margin-bottom:0.6rem;">Dataset</div>', unsafe_allow_html=True)
    dist = sim_df["scenario"].value_counts()
    for scen, count in dist.items():
        label = scen.replace("_", "/")
        st.markdown(f'<div style="font-size:0.7rem;color:rgba(232,237,228,0.6);display:flex;justify-content:space-between;padding:0.15rem 0;"><span>{label}</span><span style="color:var(--harvest);">{count}</span></div>', unsafe_allow_html=True)

    total_rows = len(sim_df)
    grain_rows = len(sim_df[sim_df["grain_type"] == grain_type]) if grain_type else total_rows
    st.markdown(f'<div style="font-size:0.65rem;color:rgba(232,237,228,0.3);margin-top:0.6rem;">Total: {total_rows} rows</div>', unsafe_allow_html=True)
    if grain_type:
        st.markdown(f'<div style="font-size:0.65rem;color:rgba(232,237,228,0.3);">{grain_type}: {grain_rows} rows</div>', unsafe_allow_html=True)


# ── Main Content ──────────────────────────────────────────────────────────
# Top header
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("""
    <h1 style="font-family:'Fraunces',serif;font-size:1.8rem;font-weight:700;color:#E8B84B;
               letter-spacing:-0.02em;margin:0;padding:0;">
        Predictive Storage Monitor
    </h1>
    <p style="font-size:0.68rem;color:rgba(232,237,228,0.4);letter-spacing:0.12em;text-transform:uppercase;margin:0.3rem 0 0;">
        Real-time threat detection · Mold · Aflatoxin · Weevils
    </p>
    """, unsafe_allow_html=True)

with col_status:
    # 1. Get the prediction from session state
    pred = st.session_state.get("last_pred", None)
    
    if pred:
        # 2. Get the risk level safely. 
        # We use .get() with () parentheses to avoid KeyErrors.
        lvl = pred.get("risk_level", "Safe")
        
        # 3. Map the risk level to the CSS color classes
        status_map = {
            "Safe": "status-safe",
            "Warning": "status-warning",
            "Critical": "status-critical",
            0: "status-safe",
            1: "status-warning",
            2: "status-critical"
        }
        
        css_class = status_map.get(lvl, "status-safe")
        
        # 4. Display the badge
        st.markdown(
            f'<div style="text-align:right;padding-top:0.5rem;">'
            f'<span class="status-badge {css_class} pulse">{lvl}</span>'
            f'</div>', 
            unsafe_allow_html=True
        )
    else:
        # Display standby if no prediction has been made yet
        st.markdown(
            '<div style="text-align:right;padding-top:0.5rem;">'
            '<span class="status-badge status-safe">Standby</span>'
            '</div>', 
            unsafe_allow_html=True
        )
st.markdown('<hr class="grain-divider">', unsafe_allow_html=True)


# ── Metric Cards ──────────────────────────────────────────────────────────
def render_metrics(temp, hum, co2):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌡️ Temperature</div>
            <div class="metric-value">{temp:.1f}</div>
            <div class="metric-unit">°C</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">💧 Humidity</div>
            <div class="metric-value">{hum:.1f}</div>
            <div class="metric-unit">%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌬️ CO₂</div>
            <div class="metric-value">{co2:.0f}</div>
            <div class="metric-unit">ppm</div>
        </div>""", unsafe_allow_html=True)


# ── Prediction Result Panel ───────────────────────────────────────────────
def render_prediction(pred: dict):
    if not pred:
        return

    # 1. First, get the raw values from the dictionary
    lvl = pred.get("risk_level", "Safe")
    conf = pred.get("confidence", 0)
    threat = pred.get("threat_type", "No immediate threat")
    scenario = pred.get("scenario", "Normal")
    
    # 2. Define 'inputs' BEFORE you use it
    inputs = pred.get("inputs", {})
    g_type = inputs.get("grain_type", "Grain")

    # 3. NOW run the Safety Override (The logic we added to fix the 'Safe' glitch)
    # This uses the 'inputs' variable we just defined above
    if inputs.get("co2_ppm", 0) > 1500 or inputs.get("humidity_pct", 0) > 80:
        lvl = "Critical"
        threat = "High Risk Conditions Detected"

    # ... the rest of your UI code (st.columns, etc.) stays the same ...

    # --- SAFE DATA FETCHING ---
    # We use .get() so if a key is missing, the app doesn't crash.
    lvl = pred.get("risk_level", "Safe")
    conf = pred.get("confidence", 0)
    threat = pred.get("threat_type", "No immediate threat")
    scenario = pred.get("scenario", "Normal")
    
    # Safely get grain type from nested inputs
    inputs = pred.get("inputs", {})
    g_type = inputs.get("grain_type", "Grain")
    
    # Get advice - handling potential missing function or keys
    try:
        advice = get_farmer_advice(scenario, g_type)
    except:
        advice = pred.get("advice", "Conditions are stable. Continue monitoring.")

    # --- UI MAPPING ---
    status_map = {
        "Safe": "status-safe", 
        "Warning": "status-warning", 
        "Critical": "status-critical",
        0: "status-safe", 1: "status-warning", 2: "status-critical"
    }
    css_class = status_map.get(lvl, "status-safe")

    # --- THE UI (FRAUNCES STYLE) ---
    c_left, c_right = st.columns([1, 2])
    
    with c_left:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;padding:1.4rem;">
            <div class="metric-label">Threat Detected</div>
            <div style="font-family:'Fraunces',serif;font-size:1.05rem;color:var(--sky);margin:0.5rem 0 0.8rem;line-height:1.35;">
                {threat}
            </div>
            <span class="status-badge {css_class}">{lvl}</span>
            <div style="margin-top:1rem;">
                <div class="metric-label">Model Confidence</div>
                <div style="font-size:1.4rem;font-family:'Fraunces',serif;color:var(--harvest);">{conf}%</div>
                <div class="conf-bar-wrap"><div class="conf-bar" style="width:{conf}%;"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown(f"""
        <div style="margin-bottom:0.4rem;font-size:0.62rem;letter-spacing:0.15em;text-transform:uppercase;color:rgba(232,237,228,0.4);">
            📋 Farmer Advisory — Sent to Telegram
        </div>
        <div class="advice-panel">{advice}</div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE: Live Simulation
# ════════════════════════════════════════════════════════════════════════════
if "Simulation" in mode:
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
    with btn_col1:
        if st.button("▶  Start Simulation"):
            st.session_state.sim_running = True
    with btn_col2:
        if st.button("⏹  Stop"):
            st.session_state.sim_running = False
    with btn_col3:
        if st.button("↺  Reset"):
            st.session_state.sim_index = 0
            st.session_state.history = []
            st.session_state.last_pred = None

    # whenever grain_type changes, restart the simulation to keep it focused
    if "sim_grain" not in st.session_state or st.session_state.sim_grain != grain_type:
        st.session_state.sim_grain = grain_type
        st.session_state.sim_index = 0
        st.session_state.history = []
        st.session_state.last_pred = None

    sim_speed = st.slider("Simulation speed (seconds per reading)", 0.5, 5.0, 1.5, 0.5)

    st.markdown('<hr class="grain-divider">', unsafe_allow_html=True)

    metrics_placeholder   = st.empty()
    prediction_placeholder = st.empty()
    log_placeholder        = st.empty()

    def run_one_step():
        # filter rows by currently selected grain; keep deterministic order
        filtered = sim_df[sim_df["grain_type"] == grain_type]
        if filtered.empty:
            # fallback to whole dataset if something went wrong
            filtered = sim_df
        idx = st.session_state.sim_index % len(filtered)
        row = filtered.iloc[idx]

        pred = predict_grain_risk(
            grain_type=str(row["grain_type"]),
            temp=float(row["temperature_c"]),
            humidity=float(row["humidity_pct"]),
            co2=float(row["co2_ppm"]),
            model=model,
            encoder=encoder,
        )

        st.session_state.last_pred = pred
        st.session_state.sim_index += 1

        # Optional Telegram alert
        if send_alerts and pred["risk_level"] != "Safe":
            send_telegram_alert(
                pred,
                bot_token=st.session_state.get("bot_token", ""),
                chat_id=st.session_state.get("chat_id", ""),
            )

        # Append to history
        st.session_state.history.append({
            "Row": idx,
            "Grain": str(row["grain_type"]),
            "Temp °C": float(row["temperature_c"]),
            "Hum %": float(row["humidity_pct"]),
            "CO₂ ppm": float(row["co2_ppm"]),
            "Status": pred["risk_level"],
            "Threat": pred["threat_type"],
        })
        if len(st.session_state.history) > 50:
            st.session_state.history = st.session_state.history[-50:]

        return pred, row

    if st.session_state.sim_running:
        pred, row = run_one_step()

        with metrics_placeholder.container():
            render_metrics(row["temperature_c"], row["humidity_pct"], row["co2_ppm"])
        with prediction_placeholder.container():
            render_prediction(pred)

        if st.session_state.history:
            with log_placeholder.container():
                st.markdown('<div style="font-size:0.62rem;letter-spacing:0.15em;text-transform:uppercase;color:rgba(232,237,228,0.4);margin-top:1rem;margin-bottom:0.5rem;">Reading History (last 50)</div>', unsafe_allow_html=True)
                hist_df = pd.DataFrame(st.session_state.history[::-1])
                st.dataframe(hist_df, use_container_width=True, height=200)

        time.sleep(sim_speed)
        st.rerun()

    elif st.session_state.last_pred:
        pred = st.session_state.last_pred
        inp = pred["inputs"]
        with metrics_placeholder.container():
            render_metrics(inp["temperature_c"], inp["humidity_pct"], inp["co2_ppm"])
        with prediction_placeholder.container():
            render_prediction(pred)

        if st.session_state.history:
            with log_placeholder.container():
                st.markdown('<div style="font-size:0.62rem;letter-spacing:0.15em;text-transform:uppercase;color:rgba(232,237,228,0.4);margin-top:1rem;margin-bottom:0.5rem;">Reading History (last 50)</div>', unsafe_allow_html=True)
                hist_df = pd.DataFrame(st.session_state.history[::-1])
                st.dataframe(hist_df, use_container_width=True, height=200)
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 0;color:rgba(232,237,228,0.3);font-size:0.8rem;letter-spacing:0.12em;">
            Press ▶ START SIMULATION to begin monitoring
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE: Manual Sensor Input
# ════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div style="font-size:0.62rem;letter-spacing:0.15em;text-transform:uppercase;color:rgba(232,237,228,0.4);margin-bottom:1rem;">Enter Sensor Readings Manually</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        temp_in = st.slider("Temperature (°C)", 10.0, 50.0, 25.0, 0.5)
    with c2:
        hum_in = st.slider("Humidity (%)", 20.0, 99.0, 60.0, 0.5)
    with c3:
        co2_in = st.slider("CO₂ (ppm)", 300.0, 4000.0, 700.0, 50.0)

    render_metrics(temp_in, hum_in, co2_in)
    st.markdown('<hr class="grain-divider">', unsafe_allow_html=True)

    if st.button("🔬 Analyse Grain Risk"):
        with st.spinner("Running ML inference…"):
            # 1. Run the prediction
            pred = predict_grain_risk(grain_type, temp_in, hum_in, co2_in, model, encoder)
            
            # 2. Store it in session state
            st.session_state.manual_pred = pred
            st.session_state.last_pred   = pred
            
            # 3. SAFE ALERT LOGIC
            # We use .get() to avoid the KeyError
            current_risk = pred.get("risk_level", "Safe")
            
            # Only send alert if it's NOT Safe and the user checked the box
            if send_alerts and current_risk != "Safe":
                try:
                    send_telegram_alert(
                        pred,
                        bot_token=st.session_state.get("bot_token", ""),
                        chat_id=st.session_state.get("chat_id", ""),
                    )
                except Exception as e:
                    st.error(f"Failed to send Telegram alert: {e}")

    if st.session_state.manual_pred:
        render_prediction(st.session_state.manual_pred)

    # if alerts are turned on but credentials are missing, show a reminder
    if send_alerts and (not st.session_state.get("bot_token") or not st.session_state.get("chat_id")):
        st.warning("Provide both Telegram token and chat ID in the sidebar to enable alerts.")


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown('<hr class="grain-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.6rem;color:rgba(232,237,228,0.2);letter-spacing:0.12em;text-transform:uppercase;padding-bottom:1rem;">
    GrainGuard AI · Built for Kenyan Smallholder Farmers · Reducing Post-Harvest Loss
</div>
""", unsafe_allow_html=True)
