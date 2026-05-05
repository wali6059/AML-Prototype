import torch

from tip_or_skip.deep_model import TabularTransformerMDN, mdn_expected_value, mdn_nll


def test_mdn_forward_shapes_and_positive_sigma():
    model = TabularTransformerMDN(
        num_numeric=3,
        cat_cardinalities=[4, 5],
        embed_dim=16,
        n_heads=4,
        n_layers=1,
        n_mix=3,
    )

    logits, pi, mu, sigma = model(torch.randn(7, 3), torch.tensor([[0, 1]] * 7))

    assert logits.shape == (7,)
    assert pi.shape == (7, 3)
    assert mu.shape == (7, 3)
    assert sigma.shape == (7, 3)
    assert torch.allclose(pi.sum(dim=1), torch.ones(7), atol=1e-5)
    assert torch.all(sigma > 0)


def test_mdn_loss_is_finite_and_expected_value_has_batch_shape():
    pi = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    mu = torch.tensor([[1.0, 2.0], [0.5, 1.5]], dtype=torch.float32)
    sigma = torch.ones_like(mu) * 0.5
    target = torch.tensor([1.5, 1.0], dtype=torch.float32)

    loss = mdn_nll(target, pi, mu, sigma)
    expected = mdn_expected_value(pi, mu)

    assert torch.isfinite(loss)
    assert expected.shape == (2,)

