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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
matplotlib.use("Agg")

from tip_or_skip.config import ARTIFACT_DIR, FIGURE_DIR, ensure_directories
from tip_or_skip.data import load_dataset
from tip_or_skip.extra import flow_features

EXPERIMENT_DIR = ARTIFACT_DIR / "experiments"


class Gcn(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(features, hidden)
        self.w2 = nn.Linear(hidden, 2)

    def forward(self, x, a):
        h = torch.relu(a @ self.w1(x))
        return a @ self.w2(h)


def _data() -> pd.DataFrame:
    return load_dataset(
        columns=[
            "time_split",
            "PULocationID",
            "DOLocationID",
            "pickup_zone",
            "dropoff_zone",
            "tip_given",
            "tip_amount",
        ]
    )


def _target(df: pd.DataFrame, split: str) -> pd.DataFrame:
    part = df[df["time_split"] == split]
    return (
        part.groupby("PULocationID")
        .agg(target_tip=("tip_amount", "mean"), target_rate=("tip_given", "mean"), target_rows=("tip_amount", "size"))
        .reset_index()
        .rename(columns={"PULocationID": "zone_id"})
    )


def _adjacency(df: pd.DataFrame, zones: list[int]) -> np.ndarray:
    idx = {zone: i for i, zone in enumerate(zones)}
    a = np.eye(len(zones), dtype=np.float32)
    edges = df.groupby(["PULocationID", "DOLocationID"]).size().reset_index(name="count")
    for _, row in edges.iterrows():
        pu = int(row["PULocationID"])
        do = int(row["DOLocationID"])
        if pu in idx and do in idx:
            weight = np.log1p(float(row["count"]))
            a[idx[pu], idx[do]] += weight
            a[idx[do], idx[pu]] += weight
    degree = a.sum(axis=1)
    inv = np.diag(1 / np.sqrt(np.maximum(degree, 1e-6)))
    return inv @ a @ inv


def _mae(pred, target, mask, col: int) -> float:
    p = pred[mask, col]
    y = target[mask, col]
    return float(torch.mean(torch.abs(p - y)).detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden", type=int, default=32)
    args = parser.parse_args()

    ensure_directories()
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    df = _data()
    train_df = df[df["time_split"] == "train"].copy()
    valid_df = df[df["time_split"] == "valid"].copy()
    test_df = df[df["time_split"] == "test"].copy()
    zones = sorted(set(df["PULocationID"].dropna().astype(int)).union(set(df["DOLocationID"].dropna().astype(int))))
    features = flow_features(train_df).set_index("zone_id").reindex(zones).fillna(0).reset_index()
    valid_target = _target(df, "valid").set_index("zone_id").reindex(zones).fillna(0)
    test_target = _target(df, "test").set_index("zone_id").reindex(zones).fillna(0)
    train_target = _target(df, "train").set_index("zone_id").reindex(zones).fillna(0)
    x_cols = ["pickup_count", "dropoff_count", "out_degree", "in_degree", "flow_count", "avg_tip", "tip_rate"]
    x = features[x_cols].to_numpy(dtype=np.float32)
    x[:, :5] = np.log1p(x[:, :5])
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std == 0] = 1
    x = (x - mean) / std
    y_train = train_target[["target_tip", "target_rate"]].to_numpy(dtype=np.float32)
    y_valid = valid_target[["target_tip", "target_rate"]].to_numpy(dtype=np.float32)
    y_test = test_target[["target_tip", "target_rate"]].to_numpy(dtype=np.float32)
    train_mask = train_target["target_rows"].to_numpy() >= 50
    valid_mask = valid_target["target_rows"].to_numpy() >= 50
    test_mask = test_target["target_rows"].to_numpy() >= 50
    a = _adjacency(train_df, zones)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Gcn(x.shape[1], args.hidden).to(device)
    xt = torch.tensor(x, device=device)
    at = torch.tensor(a, device=device)
    yt = torch.tensor(y_train, device=device)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    loss_fn = nn.L1Loss()
    history = []
    for epoch in range(args.epochs):
        pred = model(xt, at)
        loss = loss_fn(pred[train_mask_t], yt[train_mask_t])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            history.append({"epoch": epoch + 1, "train_mae": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        pred = model(xt, at).detach().cpu()
    test_t = torch.tensor(y_test)
    valid_t = torch.tensor(y_valid)
    train_t = torch.tensor(y_train)
    alphas = np.linspace(0, 1, 21)
    valid_pred = pred.numpy()
    best_alpha = 0.0
    best_valid_blend = float("inf")
    for alpha in alphas:
        blend = alpha * valid_pred[:, 0] + (1 - alpha) * y_train[:, 0]
        score = float(np.mean(np.abs(blend[valid_mask] - y_valid[valid_mask, 0])))
        if score < best_valid_blend:
            best_valid_blend = score
            best_alpha = float(alpha)
    test_blend = best_alpha * valid_pred[:, 0] + (1 - best_alpha) * y_train[:, 0]
    metrics = {
        "device": str(device),
        "nodes": int(len(zones)),
        "valid_tip_mae": _mae(pred, valid_t, valid_mask, 0),
        "test_tip_mae": _mae(pred, test_t, test_mask, 0),
        "test_tip_rate_mae": _mae(pred, test_t, test_mask, 1),
        "naive_test_tip_mae": float(np.mean(np.abs(y_train[test_mask, 0] - y_test[test_mask, 0]))),
        "naive_test_tip_rate_mae": float(np.mean(np.abs(y_train[test_mask, 1] - y_test[test_mask, 1]))),
        "blend_alpha": best_alpha,
        "blend_valid_tip_mae": best_valid_blend,
        "blend_test_tip_mae": float(np.mean(np.abs(test_blend[test_mask] - y_test[test_mask, 0]))),
    }
    out = features[["zone_id", "zone"]].copy()
    out["train_tip"] = y_train[:, 0]
    out["test_tip"] = y_test[:, 0]
    out["gcn_tip"] = pred[:, 0].numpy()
    out["blend_tip"] = test_blend
    out["test_rows"] = test_target["target_rows"].to_numpy()
    out.to_csv(EXPERIMENT_DIR / "graph_zone_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(EXPERIMENT_DIR / "graph_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(EXPERIMENT_DIR / "graph_training_history.csv", index=False)
    torch.save({"model": model.state_dict(), "zones": zones, "mean": mean, "std": std}, EXPERIMENT_DIR / "graph_gcn.pt")

    plot_df = out[out["test_rows"] >= 50].sort_values("test_rows", ascending=False).head(120)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(plot_df["test_tip"], plot_df["gcn_tip"], alpha=0.65, color="#176b54")
    lo = min(plot_df["test_tip"].min(), plot_df["gcn_tip"].min())
    hi = max(plot_df["test_tip"].max(), plot_df["gcn_tip"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#999999")
    ax.set_title("Graph Model: Zone Expected Tip")
    ax.set_xlabel("Observed 2025 zone avg tip")
    ax.set_ylabel("GCN prediction")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "graph_gcn_zone_prediction.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
