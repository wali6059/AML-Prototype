from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, model_features


@dataclass
class FeatureEncoder:
    numeric_features: list[str]
    categorical_features: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    categories: dict[str, list[str]]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "FeatureEncoder":
        features = model_features()
        numeric_features = [column for column in NUMERIC_FEATURES if column in features]
        categorical_features = [column for column in CATEGORICAL_FEATURES if column in features]
        numeric = df[numeric_features].apply(pd.to_numeric, errors="coerce")
        means = numeric.mean().fillna(0.0)
        stds = numeric.std().replace(0, 1.0).fillna(1.0)
        categories = {}
        for column in categorical_features:
            values = df[column].fillna("Unknown").astype(str)
            categories[column] = sorted(values.unique().tolist())
        return cls(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            numeric_mean=means.to_dict(),
            numeric_std=stds.to_dict(),
            categories=categories,
        )

    @property
    def cat_cardinalities(self) -> list[int]:
        return [len(self.categories[column]) + 1 for column in self.categorical_features]

    def transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        if not self.numeric_features:
            return np.zeros((len(df), 0), dtype=np.float32)
        numeric = df[self.numeric_features].apply(pd.to_numeric, errors="coerce")
        for column in self.numeric_features:
            numeric[column] = numeric[column].fillna(self.numeric_mean[column])
            numeric[column] = (numeric[column] - self.numeric_mean[column]) / self.numeric_std[column]
        return numeric.to_numpy(dtype=np.float32)

    def transform_categorical(self, df: pd.DataFrame) -> np.ndarray:
        if not self.categorical_features:
            return np.zeros((len(df), 0), dtype=np.int64)
        encoded = np.zeros((len(df), len(self.categorical_features)), dtype=np.int64)
        for idx, column in enumerate(self.categorical_features):
            lookup = {value: code + 1 for code, value in enumerate(self.categories[column])}
            values = df[column].fillna("Unknown").astype(str)
            encoded[:, idx] = values.map(lookup).fillna(0).astype(np.int64).to_numpy()
        return encoded

    def transform_for_torch(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.transform_numeric(df)),
            torch.from_numpy(self.transform_categorical(df)),
        )

    def transform_for_sklearn(self, df: pd.DataFrame) -> np.ndarray:
        return np.concatenate([self.transform_numeric(df), self.transform_categorical(df).astype(np.float32)], axis=1)


class TaxiTipDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, encoder: FeatureEncoder):
        self.x_num, self.x_cat = encoder.transform_for_torch(frame)
        self.tip_given = torch.tensor(frame["tip_given"].to_numpy(dtype=np.float32))
        self.log_tip_amount = torch.tensor(frame["log_tip_amount"].to_numpy(dtype=np.float32))
        self.tip_amount = torch.tensor(frame["tip_amount"].to_numpy(dtype=np.float32))

    def __len__(self) -> int:
        return int(len(self.tip_given))

    def __getitem__(self, idx: int):
        return self.x_num[idx], self.x_cat[idx], self.tip_given[idx], self.log_tip_amount[idx], self.tip_amount[idx]

