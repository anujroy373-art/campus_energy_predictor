import os
import pandas as pd
import numpy as np

def prepare_dataset():
    """
    Merges train_education_site1.csv, weather.csv (or weather_train.csv), and building_metadata.csv.
    Produces campus_energy_data.csv ready for feature engineering and training.
    """
    meter_file = "train_education_site1.csv"
    weather_file = "weather.csv" if os.path.exists("weather.csv") else "weather_train.csv"
    meta_file = "building_metadata.csv"

    for f in [meter_file, weather_file, meta_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file '{f}' not found in current directory.")

    print(f"Loading {meter_file}, {weather_file}, and {meta_file}...")
    meter_df = pd.read_csv(meter_file)
    weather_df = pd.read_csv(weather_file)
    meta_df = pd.read_csv(meta_file)

    # 1. Target Building 105 (Site 1)
    target_building_id = 105
    site_id = meta_df.loc[meta_df["building_id"] == target_building_id, "site_id"].iloc[0]
    site_weather = weather_df[weather_df["site_id"] == site_id].copy()

    # 2. Datetime synchronization
    meter_df["timestamp"] = pd.to_datetime(meter_df["timestamp"])
    site_weather["timestamp"] = pd.to_datetime(site_weather["timestamp"])

    # 3. Join meter telemetry with weather
    df = pd.merge(meter_df, site_weather, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 4. Clean & interpolate weather features
    temp_col = "air_temperature" if "air_temperature" in df.columns else "temperature"
    df["temperature_c"] = df[temp_col].interpolate(method="linear").bfill().ffill()

    # Derive Relative Humidity using Magnus-Tetens formula
    if "dew_temperature" in df.columns:
        T = df["temperature_c"]
        Td = df["dew_temperature"].interpolate(method="linear").bfill().ffill()
        df["humidity_pct"] = np.clip(
            100 * (np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T))),
            10.0,
            100.0
        ).round(2)
    elif "relative_humidity" in df.columns:
        df["humidity_pct"] = df["relative_humidity"].interpolate(method="linear").bfill().ffill()
    else:
        df["humidity_pct"] = 60.0

    # 5. Extract institutional calendar markers
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month

    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    # Academic Exam Weeks: Late April & Late November
    df["is_exam_week"] = (df["dayofyear"].between(110, 125) | df["dayofyear"].between(325, 340)).astype(int)
    # Semester Breaks: Summer (June-July) & Winter (Late Dec)
    df["is_vacation"] = (df["month"].isin([6, 7]) | ((df["month"] == 12) & (df["timestamp"].dt.day > 20))).astype(int)

    # 6. Target ground truth
    df["energy_consumption_kwh"] = df["meter_reading"].round(2)

    export_cols = [
        "timestamp", "temperature_c", "humidity_pct", "is_weekend",
        "is_exam_week", "is_vacation", "energy_consumption_kwh"
    ]
    output_df = df[export_cols]
    output_path = "campus_energy_data.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Dataset successfully integrated and saved to '{output_path}' ({len(output_df):,} hourly readings).")

if __name__ == "__main__":
    prepare_dataset()