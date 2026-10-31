import numpy as np
import pandas as pd

def build_features(df_input: pd.DataFrame, is_training: bool = True):
    """
    Transforms raw timestamps and telemetry into stationary, cyclical, and autoregressive features.
    """
    df = df_input.copy()
    
    # Safe datetime conversion (Python 3.12/3.13 / Arrow / NumPy compatible)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. Cyclical time encoding (preserves circular continuity)
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["dayofweek"] = df["timestamp"].dt.dayofweek

    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["sin_dayofweek"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["cos_dayofweek"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)

    # 2. Autoregressive Lag Features
    df["lag_1h"] = df["energy_consumption_kwh"].shift(1)
    df["lag_2h"] = df["energy_consumption_kwh"].shift(2)
    df["lag_24h"] = df["energy_consumption_kwh"].shift(24)
    df["lag_168h"] = df["energy_consumption_kwh"].shift(168)  # Same hour last week

    # 3. Rolling Window Aggregates (strictly shifted by 1 to prevent data leakage)
    df["rolling_mean_24h"] = df["energy_consumption_kwh"].shift(1).rolling(window=24).mean()
    df["rolling_std_24h"] = df["energy_consumption_kwh"].shift(1).rolling(window=24).std()

    if is_training:
        df = df.dropna().reset_index(drop=True)

    feature_cols = [
        "temperature_c", "humidity_pct", "is_weekend", "is_exam_week", "is_vacation",
        "sin_hour", "cos_hour", "sin_month", "cos_month", "sin_dayofweek", "cos_dayofweek",
        "lag_1h", "lag_2h", "lag_24h", "lag_168h", "rolling_mean_24h", "rolling_std_24h"
    ]

    return df, feature_cols