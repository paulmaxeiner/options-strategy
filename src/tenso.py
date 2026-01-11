# stock_lstm_pytorch_forecast_through_2027.py
# pip install yfinance torch numpy pandas matplotlib scikit-learn

import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from collections import deque


# -----------------------------
# Config
# -----------------------------
TICKER = "SPY"
START = "2015-01-01"
END = None  # None = today
TRAIN_END = "2023-12-31"  # Train only up to this date for true out-of-sample testing

LOOKBACK = 60
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
SEED = 42

FORECAST_END = "2027-12-31"  # forecast up to this date (business days)

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Data helpers
# -----------------------------
def download_data(ticker: str, start: str, end=None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError("yfinance returned no data. Check ticker/start/end and your internet connection.")
    df = df.dropna()
    if "Close" not in df.columns:
        raise RuntimeError(f"Expected 'Close' column from yfinance. Got: {list(df.columns)}")
    return df

def make_features_close_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only Close-derived features so we can generate them in the future.
    """
    out = df.copy()
    
    # Ensure Close is a Series, not DataFrame (in case of MultiIndex columns)
    close_col = out["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.squeeze()
    
    out["Return"] = close_col.pct_change()
    out["MA10"] = close_col.rolling(10).mean()
    out["MA30"] = close_col.rolling(30).mean()
    out["Vol10"] = out["Return"].rolling(10).std()
    out = out.dropna()

    # Ensure numeric
    for c in ["Close", "Return", "MA10", "MA30", "Vol10"]:
        col = out[c]
        if isinstance(col, pd.DataFrame):
            col = col.squeeze()
        out[c] = pd.to_numeric(col, errors="coerce")
    out = out.dropna()

    if len(out) < 200:
        raise RuntimeError("Not enough clean rows after feature engineering. Use earlier START date.")
    return out

def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

class SeqDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X = torch.from_numpy(X_seq)
        self.y = torch.from_numpy(y_seq).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# Model
# -----------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


# -----------------------------
# Train / eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(Xb)
    return total / len(loader.dataset)

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, trues = [], []
    for Xb, yb in loader:
        Xb = Xb.to(DEVICE)
        pred = model(Xb).cpu().numpy().reshape(-1)
        ytrue = yb.numpy().reshape(-1)
        preds.append(pred)
        trues.append(ytrue)
    return np.concatenate(preds), np.concatenate(trues)


# -----------------------------
# Stable forecasting through 2027 (no NaN/inf into scaler)
# -----------------------------
@torch.no_grad()
def forecast_through_date(
    model: nn.Module,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    df_feat: pd.DataFrame,
    feature_cols: list,
    lookback: int,
    forecast_end: str
) -> pd.Series:
    """
    Recursive forecast of next-day Close through forecast_end (business days).

    Critical: we maintain rolling features manually to guarantee finite values
    before calling x_scaler.transform(new_row).
    """
    model.eval()

    last_date = df_feat.index[-1]
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), pd.to_datetime(forecast_end))
    if len(future_dates) == 0:
        return pd.Series([], dtype=float, name="ForecastClose")

    # Need enough history to seed MA30 and Vol10
    close_col = df_feat["Close"]
    if isinstance(close_col, pd.DataFrame):
        close_col = close_col.squeeze()
    close_hist = close_col.astype(float).values.ravel()  # Ensure 1D
    
    if len(close_hist) < 31:
        raise RuntimeError("Not enough Close history to seed MA30. Use earlier START date or reduce MA windows.")

    # Rolling windows
    closes30 = deque(close_hist[-30:], maxlen=30)
    closes10 = deque(close_hist[-10:], maxlen=10)

    # Seed last 10 returns from last 11 closes
    last11 = close_hist[-11:]
    rets10 = deque([(last11[i] / last11[i - 1] - 1.0) for i in range(1, len(last11))], maxlen=10)

    prev_close = float(close_hist[-1])

    # Seed model feature window from last `lookback` actual feature rows
    X_hist_raw = df_feat[feature_cols].values
    X_hist_scaled = x_scaler.transform(X_hist_raw)
    window_scaled = X_hist_scaled[-lookback:].astype(np.float32)

    preds = []

    for dt in future_dates:
        # Predict next close from current window
        xb = torch.from_numpy(window_scaled).unsqueeze(0).to(DEVICE)  # (1, L, n_features)
        yhat_scaled = model(xb).cpu().numpy().reshape(1, 1)
        next_close = float(y_scaler.inverse_transform(yhat_scaled).reshape(-1)[0])

        # Safety: keep it finite and positive
        if (not np.isfinite(next_close)) or next_close <= 0:
            next_close = prev_close

        # Update rolling stats
        ret = next_close / prev_close - 1.0
        if not np.isfinite(ret):
            ret = 0.0

        closes30.append(next_close)
        closes10.append(next_close)
        rets10.append(ret)

        ma10 = float(np.mean(closes10))
        ma30 = float(np.mean(closes30))
        vol10 = float(np.std(list(rets10), ddof=1)) if len(rets10) >= 2 else 0.0

        if not np.isfinite(vol10):
            vol10 = 0.0

        # Build new feature row in EXACT column order
        # feature_cols must be: ["Close","Return","MA10","MA30","Vol10"]
        new_row = np.array([[next_close, ret, ma10, ma30, vol10]], dtype=float)
        new_row[~np.isfinite(new_row)] = 0.0

        # Scale and roll forward window
        new_row_scaled = x_scaler.transform(new_row).astype(np.float32)  # (1, n_features)
        window_scaled = np.vstack([window_scaled[1:], new_row_scaled])

        preds.append(next_close)
        prev_close = next_close

    return pd.Series(preds, index=future_dates, name="ForecastClose")


# -----------------------------
# Main
# -----------------------------
def main():
    df_raw = download_data(TICKER, START, END)
    df = make_features_close_only(df_raw)

    # Target: next-day Close
    df["TargetCloseNext"] = df["Close"].shift(-1)
    df = df.dropna()

    # Forecastable feature set (Close-derived only)
    feature_cols = ["Close", "Return", "MA10", "MA30", "Vol10"]

    # Final safety: drop any remaining non-finite
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df[feature_cols].values
    y = df["TargetCloseNext"].values

    # Split by date: train only up to TRAIN_END for true out-of-sample testing
    n = len(df)
    train_end_date = pd.to_datetime(TRAIN_END)
    split = len(df[df.index <= train_end_date])
    
    if split <= LOOKBACK + 10 or (n - split) <= LOOKBACK + 10:
        raise RuntimeError("Not enough rows for your LOOKBACK/TRAIN_END. Use earlier START or later TRAIN_END.")

    # Train/test split: Train = [:split], Test = [split:]
    # Model is trained ONLY on data up to TRAIN_END (true out-of-sample)
    X_train_raw, X_test_raw = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]
    
    print(f"Training period: {df.index[0]} to {df.index[split-1]}")
    print(f"Test period: {df.index[split]} to {df.index[-1]}")
    print(f"NOTE: Model trained ONLY on data through {TRAIN_END}")

    # Scale features (fit on train only - prevents data leakage)
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train_raw)
    X_test_scaled = x_scaler.transform(X_test_raw)

    # Scale target (fit on train only - prevents data leakage)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_test_scaled = y_scaler.transform(y_test_raw.reshape(-1, 1)).reshape(-1)

    # Build sequences
    X_train_seq, y_train_seq = make_sequences(X_train_scaled, y_train_scaled, LOOKBACK)
    X_test_seq, y_test_seq = make_sequences(X_test_scaled, y_test_scaled, LOOKBACK)

    train_ds = SeqDataset(X_train_seq, y_train_seq)
    test_ds = SeqDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = LSTMRegressor(n_features=len(feature_cols), hidden=64, num_layers=2, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print(f"Device: {DEVICE}")
    print(f"Rows: {len(df)} | Train seq: {len(train_ds)} | Test seq: {len(test_ds)}")

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        if epoch == 1 or epoch % 5 == 0:
            preds_scaled, trues_scaled = predict(model, test_loader)
            rmse = math.sqrt(np.mean((preds_scaled - trues_scaled) ** 2))
            print(f"Epoch {epoch:02d} | train MSE {loss:.6f} | test RMSE (scaled) {rmse:.6f}")

    # Predictions for plotting (train + test)
    train_preds_scaled, train_trues_scaled = predict(model, train_loader)
    test_preds_scaled, test_trues_scaled = predict(model, test_loader)

    train_preds = y_scaler.inverse_transform(train_preds_scaled.reshape(-1, 1)).reshape(-1)
    train_trues = y_scaler.inverse_transform(train_trues_scaled.reshape(-1, 1)).reshape(-1)
    test_preds = y_scaler.inverse_transform(test_preds_scaled.reshape(-1, 1)).reshape(-1)
    test_trues = y_scaler.inverse_transform(test_trues_scaled.reshape(-1, 1)).reshape(-1)

    # Dates aligned with sequences
    train_dates = df.index[:split][LOOKBACK:]
    test_dates = df.index[split:][LOOKBACK:]

    # Forecast through 2027-12-31 starting from last real date
    forecast_series = forecast_through_date(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        df_feat=df,
        feature_cols=feature_cols,
        lookback=LOOKBACK,
        forecast_end=FORECAST_END
    )

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(train_dates, train_trues, label="Train Actual", alpha=0.7, color='blue')
    plt.plot(train_dates, train_preds, label="Train Predicted", alpha=0.5, color='lightblue')
    plt.plot(test_dates, test_trues, label="Test Actual (2024+)", linewidth=2, color='green')
    plt.plot(test_dates, test_preds, label="Test Predicted (uses actual features - not true forecast!)", 
             linewidth=1.5, color='orange', alpha=0.7, linestyle=':')
    plt.plot(forecast_series.index, forecast_series.values, label="True Recursive Forecast to 2027", 
             linewidth=2, linestyle='--', color='red')
    
    # Add vertical line where training ends
    train_end_date = df.index[split-1]
    plt.axvline(x=train_end_date, color='purple', linestyle=':', linewidth=2, 
                label=f'Training ends ({train_end_date.strftime("%Y-%m-%d")})')
    
    # Add vertical line where real data ends and pure forecast begins
    last_real_date = df.index[-1]
    plt.axvline(x=last_real_date, color='black', linestyle=':', linewidth=2, 
                label=f'Real data ends ({last_real_date.strftime("%Y-%m-%d")})')

    plt.title(f"{TICKER}: LSTM Forecast (trained through {TRAIN_END})\nNote: 'Test Predicted' uses actual features (not true forecast). Red line is true recursive forecast.")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
