from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"

NUMERIC_FEATURES = [
    "pickup_hour", "pickup_weekday", "pickup_month",
    "trip_distance", "fare_amount", "trip_duration_minutes",
]

CATEGORICAL_FEATURES = [
    "vendor_id", "passenger_bucket", "ratecode", "store_and_fwd_flag",
    "pickup_borough", "pickup_zone", "dropoff_borough", "dropoff_zone",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TAXI_CONFIGS = {
    "yellow": {
        "pattern": "yellow_tripdata_2025-*.parquet",
        "pickup_col": "tpep_pickup_datetime",
        "dropoff_col": "tpep_dropoff_datetime",
        "extra_columns": ["Airport_fee"],
        "sample_per_file": 30000,
    },
    "green": {
        "pattern": "green_tripdata_2025-*.parquet",
        "pickup_col": "lpep_pickup_datetime",
        "dropoff_col": "lpep_dropoff_datetime",
        "extra_columns": ["trip_type", "ehail_fee"],
        "sample_per_file": 12000,
    },
}

TRAIN_MONTHS = set(range(1, 10))
VALID_MONTHS = {10}
TEST_MONTHS = {11, 12}

class TabularTransformerMDN(nn.Module):
    """
    实现 Proposal 中的架构：
    - Backbone: Tabular Transformer (Encoding features as tokens + Self-Attention)
    - Classifier Head: P(tip > 0)
    - MDN Head: Mixture Density Network for conditional tip amount
    """
    def __init__(self, num_numeric: int, cat_cardinalities: List[int], embed_dim=32, n_heads=4, n_mix=3):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embed_dim) for card in cat_cardinalities
        ])
        
        self.num_projection = nn.Linear(num_numeric, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, batch_first=True, dim_feedforward=embed_dim*2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        self.mdn_pi = nn.Linear(embed_dim, n_mix)
        self.mdn_mu = nn.Linear(embed_dim, n_mix)
        self.mdn_sigma = nn.Linear(embed_dim, n_mix)
        
    def forward(self, x_num, x_cat):
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]

        num_token = self.num_projection(x_num).unsqueeze(1)

        tokens = torch.cat([num_token] + [e.unsqueeze(1) for e in cat_embeds], dim=1)

        attn_out = self.transformer(tokens)
        pooled = attn_out.mean(dim=1)

        prob = self.classifier(pooled)

        pi = F.softmax(self.mdn_pi(pooled), dim=-1)
        mu = self.mdn_mu(pooled)
        sigma = torch.exp(self.mdn_sigma(pooled)) 
        
        return prob, pi, mu, sigma


def mdn_nll_loss(y_true, pi, mu, sigma):
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(y_true.unsqueeze(1)) 
    weighted_log_prob = log_prob + torch.log(pi + 1e-7)
    return -torch.logsumexp(weighted_log_prob, dim=1).mean()

def compute_total_loss(prob, pi, mu, sigma, y_tip_exists, y_amount, alpha=0.5):
    bce = F.binary_cross_entropy(prob, y_tip_exists.unsqueeze(1))

    mask = (y_tip_exists > 0)
    if mask.any():
        nll = mdn_nll_loss(y_amount[mask], pi[mask], mu[mask], sigma[mask])
    else:
        nll = torch.tensor(0.0, device=prob.device)
        
    return bce + alpha * nll


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
        ]
    )

def load_zone_lookup(base_dir: Path | None = None) -> pd.DataFrame:
    base_dir = base_dir or RAW_DATA_DIR
    lookup = pd.read_csv(base_dir / "taxi_zone_lookup.csv")
    return lookup.rename(columns={"LocationID": "location_id", "Borough": "borough", "Zone": "zone"})

def _add_engineered_features(df: pd.DataFrame, taxi_type: str, month: int) -> pd.DataFrame:
    pickup = pd.to_datetime(df["pickup_datetime"])
    dropoff = pd.to_datetime(df["dropoff_datetime"])
    duration = (dropoff - pickup).dt.total_seconds() / 60.0
    df = df.loc[duration > 0].copy()
    
    df["taxi_type"] = taxi_type
    df["pickup_month"] = month
    df["pickup_hour"] = pickup.loc[df.index].dt.hour.astype(int)
    df["pickup_weekday"] = pickup.loc[df.index].dt.dayofweek.astype(int)
    df["trip_duration_minutes"] = duration.loc[df.index].clip(upper=180)
    df["vendor_id"] = df["VendorID"].fillna(-1).astype(int).astype(str)
    df["ratecode"] = df["RatecodeID"].fillna(99).astype(int).astype(str)
    df["store_and_fwd_flag"] = df["store_and_fwd_flag"].fillna("N").astype(str)
    df["passenger_bucket"] = df["passenger_count"].fillna(1).clip(0, 6).astype(int).astype(str)
    df["tip_given"] = (df["tip_amount"] > 0).astype(float)
    df["log_tip_amount"] = np.log1p(df["tip_amount"].clip(lower=0))
    df["month_split"] = df["pickup_month"].map(lambda m: "train" if m < 10 else ("valid" if m == 10 else "test"))
    return df

def sample_taxi_data(taxi_type: str, base_dir: Path | None = None) -> pd.DataFrame:
    base_dir = base_dir or RAW_DATA_DIR
    config = TAXI_CONFIGS[taxi_type]
    lookup = load_zone_lookup(base_dir)
    frames = []
    
    for m, path in enumerate(sorted(base_dir.glob(config["pattern"])), start=1):
        df = pq.read_table(path).to_pandas()
        df = df.rename(columns={config["pickup_col"]: "pickup_datetime", config["dropoff_col"]: "dropoff_datetime"})
        df = _add_engineered_features(df, taxi_type, m)
        if len(df) > config["sample_per_file"]:
            df = df.sample(n=int(config["sample_per_file"]), random_state=42)
        
        df = df.merge(lookup.add_prefix("pickup_"), left_on="PULocationID", right_on="pickup_location_id", how="left") \
               .merge(lookup.add_prefix("dropoff_"), left_on="DOLocationID", right_on="dropoff_location_id", how="left")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    for col in ["pickup_borough", "pickup_zone", "dropoff_borough", "dropoff_zone"]:
        combined[col] = combined[col].fillna("Unknown")
    return combined


def train_models(df: pd.DataFrame) -> Tuple[nn.Module, ColumnTransformer, Dict]:
    train_df = df[df["month_split"].isin(["train", "valid"])]
    test_df = df[df["month_split"] == "test"]
    
    preprocessor = build_preprocessor()
    X_train_raw = preprocessor.fit_transform(train_df[MODEL_FEATURES])
    X_test_raw = preprocessor.transform(test_df[MODEL_FEATURES])

    def to_torch(raw_data):
        num = torch.FloatTensor(raw_data[:, :len(NUMERIC_FEATURES)])
        cat = torch.LongTensor(raw_data[:, len(NUMERIC_FEATURES):].astype(float) + 1) 
        return num, cat

    X_train_num, X_train_cat = to_torch(X_train_raw)
    y_train_exists = torch.FloatTensor(train_df["tip_given"].values)
    y_train_amt = torch.FloatTensor(train_df["log_tip_amount"].values)
    
    cat_cards = [len(preprocessor.transformers_[1][1].categories_[i]) for i in range(len(CATEGORICAL_FEATURES))]
    model = TabularTransformerMDN(num_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cards)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting Deep Learning training...")
    model.train()
    for epoch in range(5): 
        optimizer.zero_grad()
        prob, pi, mu, sigma = model(X_train_num, X_train_cat)
        loss = compute_total_loss(prob, pi, mu, sigma, y_train_exists, y_train_amt)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_test_num, X_test_cat = to_torch(X_test_raw)
        prob_test, pi_test, mu_test, sigma_test = model(X_test_num, X_test_cat)
        
        y_test_exists_np = test_df["tip_given"].values
        auc = roc_auc_score(y_test_exists_np, prob_test.numpy().flatten())

        expected_log_amt = torch.sum(pi_test * mu_test, dim=1).numpy()
        rmse = np.sqrt(np.mean((test_df["log_tip_amount"].values - expected_log_amt)**2))

    metrics = {
        "roc_auc": float(auc),
        "rmse_log_tip": float(rmse),
        "tip_rate_test": float(y_test_exists_np.mean()),
    }
    return model, preprocessor, metrics


def save_model_bundle(model, preprocessor, taxi_type, output_dir=None):
    output_dir = output_dir or ARTIFACT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_state": model.state_dict(),
        "preprocessor": preprocessor,
        "config": {
            "num_numeric": len(NUMERIC_FEATURES),
            "cat_cardinalities": [len(c) for c in preprocessor.transformers_[1][1].categories_]
        }
    }
    torch.save(bundle, output_dir / f"{taxi_type}_v2_model.pth")

def predict_tip(model, preprocessor, features_dict):
    model.eval()
    df = pd.DataFrame([features_dict])
    X_raw = preprocessor.transform(df[MODEL_FEATURES])
    
    num = torch.FloatTensor(X_raw[:, :len(NUMERIC_FEATURES)])
    cat = torch.LongTensor(X_raw[:, len(NUMERIC_FEATURES):].astype(float) + 1)
    
    with torch.no_grad():
        prob, pi, mu, sigma = model(num, cat)
        cond_log_amt = torch.sum(pi * mu, dim=1).item()
        cond_amt = np.expm1(cond_log_amt)
        
    return {
        "tip_probability": float(prob.item()),
        "conditional_tip": float(max(0, cond_amt)),
        "expected_tip": float(prob.item() * max(0, cond_amt))
    }

def build_summary_tables(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    monthly = df.groupby(["taxi_type", "pickup_month"]).agg(
        trips=("tip_given", "size"), tip_rate=("tip_given", "mean"), avg_tip=("tip_amount", "mean")
    ).reset_index()
    return {"monthly_summary.csv": monthly}

def save_summary_tables(df, output_dir=None):
    output_dir = output_dir or ARTIFACT_DIR
    for name, table in build_summary_tables(df).items():
        table.to_csv(output_dir / name, index=False)
    df[["pickup_zone", "pickup_borough"]].drop_duplicates().to_csv(output_dir / "zone_options.csv", index=False)

