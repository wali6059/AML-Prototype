from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
matplotlib.use("Agg")

from tip_or_skip.config import ARTIFACT_DIR, FIGURE_DIR, ensure_directories
from tip_or_skip.data import load_dataset

A_PLUS_DIR = ARTIFACT_DIR / "a_plus"


class TipLSTM(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def _hourly() -> pd.DataFrame:
    df = load_dataset(
        columns=[
            "taxi_type",
            "time_split",
            "pickup_datetime",
            "pickup_borough",
            "tip_given",
            "tip_amount",
        ]
    )
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["period"] = df["pickup_datetime"].dt.floor("h")
    hourly = (
        df.groupby(["taxi_type", "pickup_borough", "period"], observed=True)
        .agg(rows=("tip_amount", "size"), tip_rate=("tip_given", "mean"), avg_tip=("tip_amount", "mean"))
        .reset_index()
        .sort_values(["taxi_type", "pickup_borough", "period"])
    )
    hourly["hour"] = hourly["period"].dt.hour
    hourly["month"] = hourly["period"].dt.month
    hourly["split"] = np.select(
        [
            hourly["period"] < pd.Timestamp("2024-10-01"),
            hourly["period"] < pd.Timestamp("2025-01-01"),
        ],
        ["train", "valid"],
        default="test",
    )
    return hourly


def _matrix(hourly: pd.DataFrame) -> pd.DataFrame:
    out = hourly.copy()
    out["rows_log"] = np.log1p(out["rows"])
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    out["taxi_code"] = out["taxi_type"].map({"green": 0.0, "yellow": 1.0}).fillna(0.0)
    boroughs = {value: i for i, value in enumerate(sorted(out["pickup_borough"].dropna().astype(str).unique()))}
    out["borough_code"] = out["pickup_borough"].astype(str).map(boroughs).fillna(0.0)
    return out


def _windows(frame: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = ["rows_log", "tip_rate", "avg_tip", "hour_sin", "hour_cos", "month_sin", "month_cos", "taxi_code", "borough_code"]
    groups = frame.groupby(["taxi_type", "pickup_borough"], observed=True)
    xs = []
    ys = []
    splits = []
    for _, group in groups:
        group = group.sort_values("period").reset_index(drop=True)
        values = group[cols].to_numpy(dtype=np.float32)
        targets = group[["tip_rate", "avg_tip"]].to_numpy(dtype=np.float32)
        for i in range(lookback, len(group)):
            xs.append(values[i - lookback : i])
            ys.append(targets[i])
            splits.append(str(group.loc[i, "split"]))
    return np.stack(xs), np.stack(ys), splits


def _split(xs, ys, splits, name):
    idx = np.array([split == name for split in splits])
    return torch.tensor(xs[idx]), torch.tensor(ys[idx])


def _metrics(pred: np.ndarray, y: np.ndarray, naive: np.ndarray) -> dict[str, float]:
    return {
        "tip_rate_mae": float(np.mean(np.abs(pred[:, 0] - y[:, 0]))),
        "avg_tip_mae": float(np.mean(np.abs(pred[:, 1] - y[:, 1]))),
        "naive_tip_rate_mae": float(np.mean(np.abs(naive[:, 0] - y[:, 0]))),
        "naive_avg_tip_mae": float(np.mean(np.abs(naive[:, 1] - y[:, 1]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lookback", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    ensure_directories()
    A_PLUS_DIR.mkdir(parents=True, exist_ok=True)
    frame = _matrix(_hourly())
    xs, ys, splits = _windows(frame, args.lookback)
    mean = xs[np.array([s == "train" for s in splits])].mean(axis=(0, 1), keepdims=True)
    std = xs[np.array([s == "train" for s in splits])].std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1
    xs = (xs - mean) / std
    x_train, y_train = _split(xs, ys, splits, "train")
    x_valid, y_valid = _split(xs, ys, splits, "valid")
    x_test, y_test = _split(xs, ys, splits, "test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TipLSTM(x_train.shape[-1], args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    history = []
    best_state = None
    best_valid = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(x_valid.to(device)), y_valid.to(device)).detach().cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "valid_loss": valid_loss})
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x_test.to(device)).cpu().numpy()
    y = y_test.numpy()
    naive = xs[np.array([s == "test" for s in splits])][:, -1, [1, 2]]
    naive = naive * std.squeeze()[[1, 2]] + mean.squeeze()[[1, 2]]
    metrics = _metrics(pred, y, naive)
    metrics.update({"epochs": args.epochs, "lookback": args.lookback, "device": str(device), "test_rows": int(len(y))})
    pd.DataFrame(history).to_csv(A_PLUS_DIR / "sequence_training_history.csv", index=False)
    pd.DataFrame({"tip_rate_actual": y[:, 0], "tip_rate_pred": pred[:, 0], "avg_tip_actual": y[:, 1], "avg_tip_pred": pred[:, 1]}).to_csv(A_PLUS_DIR / "sequence_predictions.csv", index=False)
    (A_PLUS_DIR / "sequence_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "lookback": args.lookback}, A_PLUS_DIR / "sequence_lstm.pt")

    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(hist["epoch"], hist["train_loss"], label="train")
    ax.plot(hist["epoch"], hist["valid_loss"], label="valid")
    ax.set_title("Sequence LSTM Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "sequence_lstm_training.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
