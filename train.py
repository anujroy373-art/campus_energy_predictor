import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from prepare_data import prepare_dataset
from features import build_features

# ----------------- Deep Learning Model -----------------
class CampusLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(CampusLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

class TimeSeriesSequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=12):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],
            self.y[idx + self.seq_len]
        )

# ----------------- Evaluation Metrics -----------------
def calculate_metrics(y_true, y_pred):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"RMSE (kWh)": round(rmse, 2), "MAE (kWh)": round(mae, 2), "MAPE (%)": round(mape, 2), "R2": round(r2, 4)}

def train_and_evaluate():
    data_file = "campus_energy_data.csv"
    if not os.path.exists(data_file):
        prepare_dataset()

    print("Loading data and engineering features...")
    df_raw = pd.read_csv(data_file)
    df, feature_cols = build_features(df_raw, is_training=True)

    # Chronological Split: 80% Historical Train, 20% Future Test (Zero data leakage)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[feature_cols].values, train_df["energy_consumption_kwh"].values
    X_test, y_test = test_df[feature_cols].values, test_df["energy_consumption_kwh"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("models", exist_ok=True)
    results = {}

    # 1. Ridge Baseline
    print("\n--- Training Model 1/4: Ridge Regression (Baseline) ---")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_pred_ridge = ridge.predict(X_test_scaled)
    results["Ridge Regression"] = calculate_metrics(y_test, y_pred_ridge)
    joblib.dump(ridge, "models/ridge.joblib")

    # 2. Random Forest Regressor
    print("--- Training Model 2/4: Random Forest ---")
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest"] = calculate_metrics(y_test, y_pred_rf)
    joblib.dump(rf, "models/rf.joblib")

    # 3. XGBoost Regressor
    print("--- Training Model 3/4: XGBoost ---")
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    results["XGBoost"] = calculate_metrics(y_test, y_pred_xgb)
    joblib.dump(xgb_model, "models/xgboost.joblib")

    # 4. PyTorch LSTM Sequence Model
    print("--- Training Model 4/4: PyTorch LSTM (Sequence Length = 12 hrs) ---")
    seq_len = 12
    train_dataset = TimeSeriesSequenceDataset(X_train_scaled, y_train, seq_len=seq_len)
    test_dataset = TimeSeriesSequenceDataset(X_test_scaled, y_test, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cpu")
    lstm_model = CampusLSTM(input_dim=len(feature_cols), hidden_dim=48, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.005)

    lstm_model.train()
    epochs = 15
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = lstm_model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    lstm_model.eval()
    lstm_preds = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            out = lstm_model(batch_x)
            lstm_preds.extend(out.numpy().tolist())

    y_test_seq = y_test[seq_len:]
    results["PyTorch LSTM"] = calculate_metrics(y_test_seq, np.array(lstm_preds))
    torch.save(lstm_model.state_dict(), "models/lstm.pt")

    # Serialize artifacts
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")

    # Save benchmark table
    leaderboard = pd.DataFrame(results).T.sort_values(by="RMSE (kWh)")
    print("\n================== BENCHMARK LEADERBOARD (TEST SET) ==================")
    print(leaderboard)
    print("======================================================================")
    leaderboard.to_csv("models/benchmark_results.csv")

    # Export test sample for visualization in Streamlit
    sample_vis_df = test_df.iloc[seq_len: seq_len + 300].copy()
    sample_vis_df["pred_xgboost"] = y_pred_xgb[seq_len: seq_len + 300]
    sample_vis_df["pred_lstm"] = lstm_preds[:300]
    sample_vis_df.to_csv("models/test_inferences_sample.csv", index=False)
    print("All models and benchmark tables saved inside './models/' directory.")

if __name__ == "__main__":
    train_and_evaluate()