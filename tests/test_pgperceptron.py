"""Tests for the PGPerceptron port."""

import numpy as np
import pytest

pytest.importorskip("torch")

from automoonbot.backtest import evaluate, walk_forward
from automoonbot.baselines.pgperceptron import (
    Config,
    PGPerceptron,
    descriptors,
    feature_dim,
    reference_descriptors,
)
from automoonbot.baselines.pgperceptron.trading import Action, Position, TradeManager

from helpers import synthetic

SMALL = {
    "sma": [4, 8],
    "ema": [4, 16],
    "rsi": [4, 14],
    "zsc": [5],
    "nrm": [6],
    "grd": [7],
    "cdl": [4, 8],
    "cov": [10],
    "nts": [True],
    "ntw": [True],
    "sto": {"window": [8, 16], "k": [3, 4], "d": [3, 4]},
}


def test_descriptors_match_the_original_formulas() -> None:
    p = synthetic(400, 2, seed=3)
    f, cov = descriptors(p)
    assert f.shape == (400, 2, 122) and feature_dim() == 122
    tri = p.total_return_index()
    for i in range(2):
        k = tri[:, i] / p.close[:, i]
        ohlc = np.stack([p.open[:, i] * k, p.high[:, i] * k, p.low[:, i] * k, tri[:, i]], 1)
        for t in (299, 301, 399):
            ref = reference_descriptors(ohlc[t - 299 : t + 1], p.times[t - 299 : t + 1])
            np.testing.assert_allclose(f[t, i], ref, rtol=1e-9, atol=1e-11)
            q = tri[t - 299 : t + 1, i]
            assert cov[t, i] == pytest.approx(q.std() / q.mean(), rel=1e-12)
    assert np.isnan(f[:299]).all()
    np.testing.assert_array_equal(descriptors(p.truncate(350))[0], f[:350])


def test_descriptors_skip_halted_bars() -> None:
    p = synthetic(80, 2, halts=True)
    f, _ = descriptors(p, SMALL, queue=40)
    halted = ~p.tradable[:, 1]
    assert np.isnan(f[halted, 1]).all()
    assert np.isfinite(f[60:, 0]).all()


def test_trade_manager_confirmation_band() -> None:
    m = TradeManager(cov=0.1, alpha=1.0, gamma=0.1, leverage=1.0)  # risk = 1%
    assert m.try_buy(100.0, 0.1) == Action.Hold  # arms the band at 100
    assert m.try_buy(100.5, 0.1) == Action.Hold  # inside: re-armed at 100.5
    assert m.try_buy(101.6, 0.1) == Action.Buy  # > 100.5 * 1.01
    assert m.position == Position.Partial and m.trade.amount == 0.5
    assert m.try_sell(101.0, 0.1) == -1.0  # arms the sell band
    gain = m.try_sell(103.0, 0.1)
    assert gain == pytest.approx(1 + 0.5 * (103 / 101.6 - 1))
    assert m.position == Position.Cash and m.portfolio == pytest.approx(gain)
    with pytest.raises(ValueError):
        TradeManager(0.1, 1.0, 0.1, leverage=2.0)


def _tiny(**kw: object) -> Config:
    base = {
        "feature_params": SMALL,
        "queue_size": 40,
        "num_mem": 32,
        "mem_dim": 16,
        "episodes": 2,
        "min_episodes": 1,
        "max_training_data": 200,
    }
    base.update(kw)
    return Config(**base)  # type: ignore[arg-type]


def test_numpy_policy_matches_torch() -> None:
    import torch

    from automoonbot.baselines.pgperceptron.agent import _NumpyPolicy
    from automoonbot.baselines.pgperceptron.model import PolicyNet, draw_masks

    torch.manual_seed(0)
    net = PolicyNet(feature_dim(SMALL), num_mem=32, mem_dim=16)
    x = torch.randn(5, feature_dim(SMALL))
    p = torch.tensor([0, 1, 2, 1, 0])
    g = torch.randn(5, 1)
    pol = _NumpyPolicy(net)
    xw = pol.project(x.numpy())
    net.eval()
    ref = net(x, p, g).detach().numpy()
    got = np.stack([pol.probs(xw[i], int(p[i]), float(g[i, 0])) for i in range(5)])
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6)
    net.train()
    m = draw_masks(5, torch.Generator().manual_seed(1), torch.device("cpu"))
    ref = net(x, p, g, masks=m).detach().numpy()
    got = np.stack(
        [
            pol.probs(xw[i], int(p[i]), float(g[i, 0]), (m[0][i].numpy(), m[1][i].numpy()))
            for i in range(5)
        ]
    )
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize(
    "cfg",
    [_tiny(), _tiny(variant="repaired"), _tiny(variant="repaired", selection="validation")],
    ids=["legacy", "repaired", "validation"],
)
def test_pgperceptron_is_causal_and_deterministic(cfg: Config) -> None:
    p = synthetic(420, 1, seed=5)
    folds = list(
        walk_forward(420, test_size=30, min_train=150, max_label_horizon=1, train_window=200)
    )
    a = evaluate(p, PGPerceptron(cfg, seed=1), folds, audit=4)
    b = evaluate(p, PGPerceptron(cfg, seed=1), folds)
    np.testing.assert_array_equal(a.log_returns, b.log_returns)
    assert np.isfinite(a.log_returns).all()
