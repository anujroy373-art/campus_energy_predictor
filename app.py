import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure modern executive layout
st.set_page_config(
    page_title="Smart Campus Energy Intelligence System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for executive engineering dashboard look
st.markdown("""
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        color: #F8FAFC;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .badge-tag {
        display: inline-block;
        background: #1E293B;
        color: #38BDF8;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        border: 1px solid #334155;
    }
    .metric-card {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1E293B;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifacts():
    if not os.path.exists("models/xgboost.joblib"):
        st.error("Model artifacts not found! Please run 'python train.py' first.")
        st.stop()
    xgb_model = joblib.load("models/xgboost.joblib")
    rf_model = joblib.load("models/rf.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    leaderboard = pd.read_csv("models/benchmark_results.csv", index_col=0)
    sample_df = pd.read_csv("models/test_inferences_sample.csv")
    return xgb_model, rf_model, feature_cols, leaderboard, sample_df

xgb_model, rf_model, feature_cols, leaderboard, sample_df = load_artifacts()

# Sidebar: System Specifications & Architecture Telemetry
with st.sidebar:
    st.markdown("### System Telemetry")
    st.markdown("""
    <div>
        <span class="badge-tag">ASHRAE Site 1</span>
        <span class="badge-tag">Bldg 105: Education</span>
        <span class="badge-tag">50,623 sq ft</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Production Pipeline Details**")
    st.markdown("""
    * **Temporal Granularity:** Hourly (1-Hour Horizon)
    * **Feature Engineering:**
      * Diurnal & Weekly $\\sin / \\cos$ cycles
      * Autoregressive Lags ($t-1, t-2, t-24, t-168$)
      * 24h Moving Mean & Volatility
    * **Validation Strategy:** Chronological Forward Split (Zero Leakage)
    * **Substation Ceiling:** 120.0 kWh (Max Sanctioned)
    """)
    st.markdown("---")
    st.caption("Engineered for Substation Load Shaving, Peak Billing Mitigation, and HVAC Scheduling.")

# Main Dashboard Header
st.markdown('<div class="main-title">Smart Campus Energy Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated 1-Hour Horizon Load Forecasting, Multi-Model Benchmarking, and Anomaly Detection on Academic Building Telemetry</div>', unsafe_allow_html=True)

# Overview Metric Highlights
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown('<div class="metric-card"><div class="metric-label">Best Performing Model</div><div class="metric-value">Random Forest</div></div>', unsafe_allow_html=True)
with col_m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Top Test R² Score</div><div class="metric-value">{leaderboard.loc["Random Forest", "R2"]:.4f}</div></div>', unsafe_allow_html=True)
with col_m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Test RMSE</div><div class="metric-value">{leaderboard.loc["Random Forest", "RMSE (kWh)"]:.2f} <span style="font-size:1rem;color:#94A3B8;">kWh</span></div></div>', unsafe_allow_html=True)
with col_m4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Test MAPE</div><div class="metric-value">{leaderboard.loc["Random Forest", "MAPE (%)"]:.2f}%</div></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "Model Evaluation & Benchmark",
    "Real-Time Load Simulator",
    "Operational Anomaly Monitor"
])

with tab1:
    st.markdown("#### Chronological Out-of-Time Benchmark")
    st.caption("Models trained chronologically on past historical sequences and evaluated on strictly unseen out-of-time telemetry.")
    
    formatted_leaderboard = leaderboard.copy()
    for col in ["RMSE (kWh)", "MAE (kWh)", "MAPE (%)"]:
        if col in formatted_leaderboard.columns:
            formatted_leaderboard[col] = formatted_leaderboard[col].map("{:.2f}".format)
    if "R2" in formatted_leaderboard.columns:
        formatted_leaderboard["R2"] = formatted_leaderboard["R2"].map("{:.4f}".format)

    st.dataframe(formatted_leaderboard, use_container_width=True)

    st.markdown("#### Empirical vs Predicted Demand Curve (300-Hour Evaluation Window)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample_df["timestamp"], y=sample_df["energy_consumption_kwh"],
        name="Actual Ground Truth (kWh)",
        line=dict(color="#38BDF8", width=2.2)
    ))
    fig.add_trace(go.Scatter(
        x=sample_df["timestamp"], y=sample_df["pred_xgboost"],
        name="XGBoost Regressor",
        line=dict(color="#10B981", dash="dash", width=1.8)
    ))
    fig.add_trace(go.Scatter(
        x=sample_df["timestamp"], y=sample_df["pred_lstm"],
        name="PyTorch LSTM Sequence Model",
        line=dict(color="#F59E0B", dash="dot", width=1.6)
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0A0F1D",
        plot_bgcolor="#0A0F1D",
        xaxis=dict(title="Timestamp (Hourly)", gridcolor="#1E293B", showgrid=True),
        yaxis=dict(title="Energy Demand (kWh)", gridcolor="#1E293B", showgrid=True),
        hovermode="x unified",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Real-Time Scenario Parameterization")
    st.caption("Simulate environmental conditions and institutional scheduling dynamics to assess prospective 1-hour load profiles.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Temporal Parameters**")
        hour_val = st.slider("Hour of Day (24-Hour)", 0, 23, 14)
        month_val = st.slider("Calendar Month", 1, 12, 10)
        day_val = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=2
        )
        dayofweek_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_val)

    with col2:
        st.markdown("**Environmental Metrics**")
        temp_val = st.slider("Ambient Temperature (°C)", -5.0, 45.0, 24.5, step=0.5)
        humidity_val = st.slider("Relative Humidity (%)", 10, 100, 60)

    with col3:
        st.markdown("**Operational State**")
        is_weekend = 1 if dayofweek_idx in [5, 6] else 0
        is_exam = st.checkbox("Active Examination Period", value=False)
        is_vacation = st.checkbox("Academic Recess / Vacation", value=False)
        prev_1h = st.number_input("Preceding Hour Load (t-1 kWh)", value=82.0, step=2.0)
        prev_24h = st.number_input("Prior Day Concurrent Load (t-24 kWh)", value=80.0, step=2.0)

    # Cyclical trigonometric transforms
    sin_hour = np.sin(2 * np.pi * hour_val / 24.0)
    cos_hour = np.cos(2 * np.pi * hour_val / 24.0)
    sin_month = np.sin(2 * np.pi * month_val / 12.0)
    cos_month = np.cos(2 * np.pi * month_val / 12.0)
    sin_day = np.sin(2 * np.pi * dayofweek_idx / 7.0)
    cos_day = np.cos(2 * np.pi * dayofweek_idx / 7.0)

    input_vector = np.array([[
        temp_val, humidity_val, is_weekend, int(is_exam), int(is_vacation),
        sin_hour, cos_hour, sin_month, cos_month, sin_day, cos_day,
        prev_1h, prev_1h * 0.98, prev_24h, prev_24h * 1.02,
        prev_24h, 12.0
    ]])

    pred_load = float(xgb_model.predict(input_vector)[0])

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(label="Predicted 1-Hour Horizon Demand", value=f"{pred_load:.2f} kWh")
        cost_est = pred_load * 8.50
        st.caption(f"Estimated Commercial Billing: ₹{cost_est:,.2f} / hr (@ ₹8.50/kWh)")

    with res_col2:
        if pred_load > 120.0:
            st.error("CRITICAL PEAK LOAD WARNING: Projected consumption breaches sanctioned contract ceiling (120.0 kWh). Stagger chiller staging or defer laboratory heavy machinery.")
        elif pred_load < 62.0:
            st.info("BASELOAD OPERATION: Building operating at minimal nocturnal/weekend standby levels.")
        else:
            st.success("NOMINAL OPERATIONAL RANGE: Demand profile aligns with typical operational boundaries.")

with tab3:
    st.markdown("#### Statistical Residual Anomaly Inspector")
    st.caption("Identifies off-hour equipment leakages and unmonitored surges where empirical load exceeds statistical expectation (|Actual - Forecast| > 2.5σ).")

    sample_df["residual"] = sample_df["energy_consumption_kwh"] - sample_df["pred_xgboost"]
    sigma_threshold = sample_df["residual"].std() * 2.5
    anomalies_detected = sample_df[sample_df["residual"] > sigma_threshold].copy()

    st.markdown(f"**Detected Anomalous Outliers:** `{len(anomalies_detected)}` instances in evaluation sample")
    if not anomalies_detected.empty:
        anomalies_display = anomalies_detected[["timestamp", "energy_consumption_kwh", "pred_xgboost", "residual", "temperature_c"]].copy()
        anomalies_display.columns = [
            "Timestamp", "Actual Telemetry (kWh)", "Model Forecast (kWh)",
            "Unplanned Surge (kWh)", "Ambient Temp (°C)"
        ]
        for col in ["Actual Telemetry (kWh)", "Model Forecast (kWh)", "Unplanned Surge (kWh)"]:
            anomalies_display[col] = anomalies_display[col].map("{:.2f}".format)
        st.dataframe(anomalies_display, use_container_width=True)