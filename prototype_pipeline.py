from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR / "data" 
ARTIFACT_DIR = ROOT_DIR / "artifacts"

NUMERIC_FEATURES = ["pickup_hour", "pickup_weekday", "pickup_month", "trip_distance", "fare_amount", "trip_duration_minutes"]
CATEGORICAL_FEATURES = ["vendor_id", "passenger_bucket", "ratecode", "store_and_fwd_flag", "pickup_borough", "pickup_zone", "dropoff_borough", "dropoff_zone"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- 通用转换工具 (放在最外层防止 NameError) ---
def to_torch(raw):
    """将 numpy 阵列安全转换为 PyTorch 张量"""
    # 使用 .copy() 修复 "non-writable" 警告
    num = torch.FloatTensor(raw[:, :len(NUMERIC_FEATURES)].copy())
    cat = torch.LongTensor(raw[:, len(NUMERIC_FEATURES):].astype(float).copy() + 1)
    return num, cat

# --- Transformer + MDN 架构 ---
class TabularTransformerMDN(nn.Module):
    def __init__(self, num_numeric: int, cat_cardinalities: List[int], embed_dim=32, n_heads=4, n_mix=3):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card + 1, embed_dim) for card in cat_cardinalities])
        self.num_projection = nn.Linear(num_numeric, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True, dim_feedforward=embed_dim*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())
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

def compute_total_loss(prob, pi, mu, sigma, y_tip_exists, y_amount, alpha=0.5):
    bce = F.binary_cross_entropy(prob, y_tip_exists.unsqueeze(1))
    mask = (y_tip_exists.squeeze() > 0)
    if mask.any():
        m = torch.distributions.Normal(loc=mu[mask], scale=sigma[mask])
        log_prob = m.log_prob(y_amount[mask].unsqueeze(1))
        weighted_log_prob = log_prob + torch.log(pi[mask] + 1e-7)
        nll = -torch.logsumexp(weighted_log_prob, dim=1).mean()
    else:
        nll = torch.tensor(0.0)
    return bce + alpha * nll

def sample_taxi_data(taxi_type: str) -> pd.DataFrame:
    file_path = RAW_DATA_DIR / "tip_or_skip_final_dataset.parquet"
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(file_path)
    if 'taxi_type' in df.columns:
        df = df[df['taxi_type'].str.lower() == taxi_type.lower()].copy()
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42)
    if 'month_split' not in df.columns and 'pickup_month' in df.columns:
        df["month_split"] = df["pickup_month"].map(lambda m: "train" if m < 10 else ("valid" if m == 10 else "test"))
    return df.reset_index(drop=True)

def train_models(df: pd.DataFrame) -> Tuple[nn.Module, ColumnTransformer, Dict]:
    train_df = df[df["month_split"].isin(["train", "valid"])]
    test_df = df[df["month_split"] == "test"]
    preprocessor = ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES), ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES)])
    
    print("Fitting preprocessor...")
    X_train_raw = preprocessor.fit_transform(train_df[MODEL_FEATURES])
    X_tr_num, X_tr_cat = to_torch(X_train_raw)
    y_tr_ex = torch.FloatTensor(train_df["tip_given"].values.copy())
    y_tr_am = torch.FloatTensor(train_df["log_tip_amount"].values.copy())
    
    dataset = TensorDataset(X_tr_num, X_tr_cat, y_tr_ex, y_tr_am)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    
    cat_cards = [len(preprocessor.transformers_[1][1].categories_[i]) for i in range(len(CATEGORICAL_FEATURES))]
    model = TabularTransformerMDN(num_numeric=len(NUMERIC_FEATURES), cat_cardinalities=cat_cards)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(5):
        model.train()
        total_loss = 0
        for b_num, b_cat, b_ex, b_am in loader:
            optimizer.zero_grad()
            prob, pi, mu, sigma = model(b_num, b_cat)
            loss = compute_total_loss(prob, pi, mu, sigma, b_ex, b_am)
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/5 - Avg Loss: {total_loss/len(loader):.4f}")
    
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        X_test_raw = preprocessor.transform(test_df[MODEL_FEATURES])
        X_te_num, X_te_cat = to_torch(X_test_raw) # 此时调用不会再报错 NameError
        p_te, pi_te, mu_te, _ = model(X_te_num, X_te_cat)
        
        y_test_ex = test_df["tip_given"].values
        auc = roc_auc_score(y_test_ex, p_te.numpy().flatten())
        rmse = np.sqrt(np.mean((test_df["log_tip_amount"].values - torch.sum(pi_te * mu_te, dim=1).numpy())**2))
    
    # 补全 metrics 以修复 app.py 的 KeyError
    metrics = {
        "roc_auc": float(auc),
        "rmse_log_tip": float(rmse),
        "tip_rate_test": float(y_test_ex.mean())
    }
    return model, preprocessor, metrics

def predict_tip(bundle: dict, feature_values: dict) -> dict:
    model = bundle["model"]; preprocessor = bundle["preprocessor"]
    model.eval()
    row = pd.DataFrame([feature_values])
    X_raw = preprocessor.transform(row[MODEL_FEATURES])
    num, cat = to_torch(X_raw)
    with torch.no_grad():
        prob, pi, mu, _ = model(num, cat)
        cond_amt = np.expm1(torch.sum(pi * mu, dim=1).item())
    return {"tip_probability": float(prob.item()), "conditional_tip": float(max(0, cond_amt)), "expected_tip": float(prob.item() * max(0, cond_amt))}