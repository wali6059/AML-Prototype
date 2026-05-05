from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from .features import FeatureEncoder
from .metrics import classification_summary, regression_summary


@dataclass
class HurdleBaseline:
    name: str
    encoder: FeatureEncoder
    classifier: object
    regressor: object

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self.encoder.transform_for_sklearn(df)
        tip_probability = self.classifier.predict_proba(x)[:, 1]
        log_tip_mean = self.regressor.predict(x)
        conditional_tip_mean = np.expm1(log_tip_mean).clip(min=0)
        return pd.DataFrame(
            {
                "tip_probability": tip_probability,
                "conditional_log_tip_mean": log_tip_mean,
                "conditional_tip_mean": conditional_tip_mean,
                "expected_tip": tip_probability * conditional_tip_mean,
            },
            index=df.index,
        )


def fit_baselines(train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, HurdleBaseline]:
    train_valid = pd.concat([train, valid], ignore_index=True)
    encoder = FeatureEncoder.fit(train_valid)
    x = encoder.transform_for_sklearn(train_valid)
    y = train_valid["tip_given"].to_numpy(dtype=int)
    tipped = train_valid["tip_given"] == 1

    logistic = LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1)
    logistic.fit(x, y)
    ridge = Ridge(alpha=1.0)
    ridge.fit(x[tipped.to_numpy()], train_valid.loc[tipped, "log_tip_amount"].to_numpy(dtype=float))

    tree_classifier = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.06,
        l2_regularization=0.02,
        random_state=42,
        early_stopping=True,
    )
    tree_classifier.fit(x, y)
    tree_regressor = HistGradientBoostingRegressor(
        max_iter=220,
        learning_rate=0.06,
        l2_regularization=0.02,
        random_state=42,
        early_stopping=True,
    )
    tree_regressor.fit(x[tipped.to_numpy()], train_valid.loc[tipped, "log_tip_amount"].to_numpy(dtype=float))

    return {
        "logistic_ridge": HurdleBaseline("logistic_ridge", encoder, logistic, ridge),
        "tree_hurdle": HurdleBaseline("tree_hurdle", encoder, tree_classifier, tree_regressor),
    }


def evaluate_baseline(model: HurdleBaseline, test: pd.DataFrame) -> dict[str, float]:
    pred = model.predict_frame(test)
    class_metrics = classification_summary(test["tip_given"], pred["tip_probability"])
    tipped = test["tip_given"] == 1
    reg_metrics = regression_summary(
        test.loc[tipped, "log_tip_amount"],
        pred.loc[tipped, "conditional_log_tip_mean"],
    )
    return {
        **{f"class_{key}": value for key, value in class_metrics.items()},
        **{f"logtip_{key}": value for key, value in reg_metrics.items()},
        "expected_tip_mae": regression_summary(test["tip_amount"], pred["expected_tip"])["mae"],
    }


def save_baselines(models: dict[str, HurdleBaseline], output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, output_dir / f"{name}.joblib")

