from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .deep_model import TabularTransformerMDN, combined_hurdle_loss, mdn_expected_value, mdn_nll
from .features import FeatureEncoder, TaxiTipDataset
from .metrics import classification_summary, interval_coverage, regression_summary


@dataclass
class TrainConfig:
    epochs: int = 18
    batch_size: int = 4096
    learning_rate: float = 1e-3
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_mix: int = 5
    dropout: float = 0.1
    mdn_weight: float = 0.7
    patience: int = 4
    seed: int = 42
    train_sample: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(frame: pd.DataFrame, encoder: FeatureEncoder, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TaxiTipDataset(frame, encoder), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict_deep_model(
    model: TabularTransformerMDN,
    encoder: FeatureEncoder,
    frame: pd.DataFrame,
    batch_size: int = 8192,
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    model.eval()
    device = torch.device(device)
    loader = _loader(frame, encoder, batch_size=batch_size, shuffle=False)
    rows = []
    with torch.no_grad():
        for x_num, x_cat, _, _, _ in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            logits, pi, mu, sigma = model(x_num, x_cat)
            prob = torch.sigmoid(logits)
            expected_log = mdn_expected_value(pi, mu)
            normal = torch.distributions.Normal(mu, sigma)
            samples = normal.sample((80,))
            components = torch.distributions.Categorical(pi).sample((80,))
            chosen = samples.gather(2, components.unsqueeze(-1)).squeeze(-1)
            tip_samples = torch.expm1(chosen).clamp_min(0)
            q10 = torch.quantile(tip_samples, 0.10, dim=0)
            q50 = torch.quantile(tip_samples, 0.50, dim=0)
            q90 = torch.quantile(tip_samples, 0.90, dim=0)
            cond_tip = torch.expm1(expected_log).clamp_min(0)
            rows.append(
                pd.DataFrame(
                    {
                        "tip_probability": prob.cpu().numpy(),
                        "conditional_log_tip_mean": expected_log.cpu().numpy(),
                        "conditional_tip_mean": cond_tip.cpu().numpy(),
                        "expected_tip": (prob * cond_tip).cpu().numpy(),
                        "q10_tip": q10.cpu().numpy(),
                        "q50_tip": q50.cpu().numpy(),
                        "q90_tip": q90.cpu().numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def evaluate_deep_model(
    model: TabularTransformerMDN,
    encoder: FeatureEncoder,
    frame: pd.DataFrame,
    device: str | torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    truth = frame.reset_index(drop=True)
    predictions = predict_deep_model(model, encoder, truth, device=device)
    class_metrics = classification_summary(truth["tip_given"], predictions["tip_probability"])
    tipped = truth["tip_given"] == 1
    reg_metrics = regression_summary(
        truth.loc[tipped, "log_tip_amount"],
        predictions.loc[tipped, "conditional_log_tip_mean"],
    )
    interval_metrics = interval_coverage(
        truth.loc[tipped, "tip_amount"],
        predictions.loc[tipped, "q10_tip"],
        predictions.loc[tipped, "q90_tip"],
    )
    metrics = {
        **{f"class_{key}": value for key, value in class_metrics.items()},
        **{f"logtip_{key}": value for key, value in reg_metrics.items()},
        **{f"interval80_{key}": value for key, value in interval_metrics.items()},
        "expected_tip_mae": regression_summary(truth["tip_amount"], predictions["expected_tip"])["mae"],
    }
    return metrics, predictions


def train_transformer_mdn(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    output_dir: Path,
    config: TrainConfig,
) -> tuple[TabularTransformerMDN, FeatureEncoder, dict[str, object]]:
    set_seed(config.seed)
    if config.train_sample is not None and len(train) > config.train_sample:
        train = train.sample(n=config.train_sample, random_state=config.seed).reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FeatureEncoder.fit(train)
    train_loader = _loader(train, encoder, config.batch_size, shuffle=True)
    valid_loader = _loader(valid, encoder, config.batch_size, shuffle=False)
    model = TabularTransformerMDN(
        num_numeric=len(encoder.numeric_features),
        cat_cardinalities=encoder.cat_cardinalities,
        embed_dim=config.embed_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        n_mix=config.n_mix,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    positive_rate = float(train["tip_given"].mean())
    positive_weight = (1.0 - positive_rate) / max(positive_rate, 1e-6)
    history = []
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for x_num, x_cat, tip_given, log_tip, _ in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            tip_given = tip_given.to(device)
            log_tip = log_tip.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, pi, mu, sigma = model(x_num, x_cat)
            loss = combined_hurdle_loss(
                logits,
                pi,
                mu,
                sigma,
                tip_given,
                log_tip,
                mdn_weight=config.mdn_weight,
                positive_weight=positive_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for x_num, x_cat, tip_given, log_tip, _ in valid_loader:
                x_num = x_num.to(device)
                x_cat = x_cat.to(device)
                tip_given = tip_given.to(device)
                log_tip = log_tip.to(device)
                logits, pi, mu, sigma = model(x_num, x_cat)
                valid_losses.append(
                    float(
                        combined_hurdle_loss(
                            logits,
                            pi,
                            mu,
                            sigma,
                            tip_given,
                            log_tip,
                            mdn_weight=config.mdn_weight,
                            positive_weight=positive_weight,
                        )
                        .detach()
                        .cpu()
                    )
                )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "valid_loss": float(np.mean(valid_losses)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["valid_loss"] < best_loss:
            best_loss = row["valid_loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "num_numeric": len(encoder.numeric_features),
            "cat_cardinalities": encoder.cat_cardinalities,
        },
        output_dir / "transformer_mdn.pt",
    )
    joblib.dump(encoder, output_dir / "feature_encoder.joblib")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    return model, encoder, {"device": str(device), "history": history, "best_valid_loss": best_loss}
