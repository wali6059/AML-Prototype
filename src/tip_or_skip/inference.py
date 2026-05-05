from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .deep_model import TabularTransformerMDN
from .training import predict_deep_model


def load_deep_bundle(artifact_dir: Path):
    checkpoint = torch.load(artifact_dir / "transformer_mdn.pt", map_location="cpu")
    encoder = joblib.load(artifact_dir / "feature_encoder.joblib")
    model = TabularTransformerMDN(
        num_numeric=checkpoint["num_numeric"],
        cat_cardinalities=checkpoint["cat_cardinalities"],
        embed_dim=checkpoint["config"]["embed_dim"],
        n_heads=checkpoint["config"]["n_heads"],
        n_layers=checkpoint["config"]["n_layers"],
        n_mix=checkpoint["config"]["n_mix"],
        dropout=checkpoint["config"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, encoder


def predict_row(model, encoder, row: dict[str, object]) -> dict[str, float]:
    predictions = predict_deep_model(model, encoder, pd.DataFrame([row]))
    out = predictions.iloc[0].to_dict()
    return {key: float(value) if isinstance(value, (np.floating, float)) else value for key, value in out.items()}

