from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class TabularTransformerMDN(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list[int],
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_mix: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities
        self.n_mix = n_mix
        self.numeric_projection = nn.Linear(num_numeric, embed_dim)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality + 2, embed_dim) for cardinality in cat_cardinalities]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, 1)
        self.mdn_pi = nn.Linear(embed_dim, n_mix)
        self.mdn_mu = nn.Linear(embed_dim, n_mix)
        self.mdn_sigma = nn.Linear(embed_dim, n_mix)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        tokens = [self.numeric_projection(x_num).unsqueeze(1)]
        for idx, embedding in enumerate(self.cat_embeddings):
            tokens.append(embedding(x_cat[:, idx]).unsqueeze(1))
        encoded = self.encoder(torch.cat(tokens, dim=1))
        pooled = self.norm(encoded.mean(dim=1))
        logits = self.classifier(pooled).squeeze(-1)
        pi = F.softmax(self.mdn_pi(pooled), dim=-1)
        mu = self.mdn_mu(pooled)
        sigma = F.softplus(self.mdn_sigma(pooled)) + 1e-4
        return logits, pi, mu, sigma


def mdn_nll(y_true: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    y_true = y_true.unsqueeze(1)
    normal = torch.distributions.Normal(mu, sigma)
    log_prob = normal.log_prob(y_true) + torch.log(pi.clamp_min(1e-8))
    return -torch.logsumexp(log_prob, dim=1).mean()


def mdn_expected_value(pi: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return torch.sum(pi * mu, dim=1)


def combined_hurdle_loss(
    logits: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    tip_given: torch.Tensor,
    log_tip_amount: torch.Tensor,
    mdn_weight: float = 0.7,
    positive_weight: float | None = None,
) -> torch.Tensor:
    pos_weight = None
    if positive_weight is not None:
        pos_weight = torch.tensor(positive_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, tip_given.float(), pos_weight=pos_weight)
    mask = tip_given > 0
    if mask.any():
        nll = mdn_nll(log_tip_amount[mask].float(), pi[mask], mu[mask], sigma[mask])
    else:
        nll = torch.zeros((), device=logits.device)
    return bce + mdn_weight * nll

