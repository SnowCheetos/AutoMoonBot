"""Tests for the graph transformer: features, universe graph, token layout,
model invariants, the training objective and the walk-forward strategy."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# Tiny tensors: threading only adds overhead (and contends with other jobs).
torch.set_num_threads(1)

from automoonbot.backtest.evaluate import check_causal  # noqa: E402
from automoonbot.models.data import PanelData, TokenLayout  # noqa: E402
from automoonbot.models.features import CHANNELS, price_features  # noqa: E402
from automoonbot.models.strategy import (  # noqa: E402
    GraphTransformerStrategy,
    Overlay,
    TrainConfig,
    cvar,
    drift,
)
from automoonbot.models.transformer import (  # noqa: E402
    GraphTransformer,
    ModelConfig,
    student_t_nll,
)
from automoonbot.models.universe import Membership, build_universe_graph  # noqa: E402

from helpers import synthetic  # noqa: E402

T, S = 260, 10


def universe(seed: int = 1, stagger: int = 12):  # type: ignore[no-untyped-def]
    p = synthetic(T, S, seed=seed)
    first = np.minimum(np.arange(S) * stagger, T - 1)
    mem = [Membership(sec, f"ind{i % 4}", f"sec{i % 2}") for i, sec in enumerate(p.securities)]
    return p, build_universe_graph(mem, p.decision_time[first]), first


def small_model(n_classes: int = 6) -> GraphTransformer:
    torch.manual_seed(0)
    cfg = ModelConfig(
        channels=len(CHANNELS),
        window=16,
        patch=4,
        d_model=16,
        heads=2,
        graph_layers=2,
        n_classes=n_classes,
        dropout=0.0,
        max_weight=0.3,
    )
    return GraphTransformer(cfg).eval()


def test_features_are_causal_and_bounded() -> None:
    p, _, _ = universe()
    full = price_features(p)
    cut = price_features(p.truncate(150))
    np.testing.assert_array_equal(full.values[:150], cut.values)
    np.testing.assert_array_equal(full.ok[:150], cut.ok)
    v = full.values[full.ok]
    assert np.isfinite(v).all() and np.abs(v).max() <= 5.0
    assert full.values.shape == (T, S, len(CHANNELS))


def test_layout_follows_knowledge_time() -> None:
    p, g, first = universe()
    lay = TokenLayout(g, p.securities)
    assert lay.n_tokens == S + 4 + 2
    early = lay.day(int(p.decision_time[first[1]]))  # two companies known
    # Classes ind0, ind1, sec0, sec1 are known; ind2, ind3 are not.
    known = {k for k, ok in zip(lay.class_keys, early.class_known, strict=True) if ok}
    assert known == {
        (1, "industry:ind0"),
        (1, "industry:ind1"),
        (2, "sector:sec0"),
        (2, "sector:sec1"),
    }
    src, _, et = early.edges
    assert set(src[et == 0].tolist()) == {0, 1}  # in_industry from known companies only
    late = lay.day(int(p.decision_time[-1]))
    assert late.class_known.all() and late.edges.shape[1] == 2 * (S + 4)
    # Vocabulary at a time only covers classes known then.
    assert len(lay.vocabulary(int(p.decision_time[first[1]]))) == 4


def test_sub_universe_remaps_edges() -> None:
    p, g, _ = universe()
    lay = TokenLayout(g, p.securities)
    t = int(p.decision_time[-1])
    cols = torch.tensor([1, 4, 7])
    ok = torch.ones(1, 3, dtype=torch.bool)
    edges, key_ok, kind = lay.batch([t], ok, cols)
    assert key_ok.shape == (1, 3 + 6) and kind.tolist()[:3] == [0, 0, 0]
    # Company 4 (new index 1) is in industry ind0 (token S + 0 -> 3 + 0).
    _, src, dst, et = edges.numpy()
    ind0 = 3 + lay.class_keys.index((1, "industry:ind0"))
    assert ((src == 1) & (dst == ind0) & (et == 0)).any()
    assert src.max() < 9 and dst.max() < 9


def encode_day(net: GraphTransformer, data: PanelData, lay: TokenLayout, p, t: int):  # type: ignore[no-untyped-def]
    days = torch.tensor([t])
    x, present, pair, valid = data.windows(days, net.cfg.window)
    edges, key_ok, kind = lay.batch([int(p.decision_time[t])], valid)
    e = net.edges(kind.numel(), edges, key_ok)
    ids = lay.class_ids(lay.vocabulary(int(p.decision_time[-1])))
    return net.encode(x, present, pair, valid, kind, ids, e), valid


def test_absent_companies_do_not_influence_others() -> None:
    p, g, first = universe()
    lay = TokenLayout(g, p.securities)
    net = small_model()
    full = PanelData.build(p, (1, 5))
    ok = full.ok.clone()
    ok[:, 6:] = False  # companies 6..9 have no data
    data = PanelData(
        full.feats, ok, full.intraday, full.close_ret, full.r1, full.labels, full.label_ok
    )
    t = int(first[-1]) + 30
    (z, preds), valid = encode_day(net, data, lay, p, t)
    assert valid[0, :6].all() and not valid[0, 6:].any()
    # Garbage features for an absent company and for the future change nothing.
    feats = data.feats.clone()
    feats[8] = 123.0
    feats[:, t + 1 :] = -7.0
    data2 = PanelData(feats, ok, data.intraday, data.close_ret, data.r1, data.labels, data.label_ok)
    (z2, preds2), _ = encode_day(net, data2, lay, p, t)
    torch.testing.assert_close(z[:, :6], z2[:, :6], rtol=0, atol=0)
    torch.testing.assert_close(preds[:, :6], preds2[:, :6], rtol=0, atol=0)
    # Present companies do influence each other (through attention).
    feats[2, : t + 1] = 3.0
    data3 = PanelData(feats, ok, data.intraday, data.close_ret, data.r1, data.labels, data.label_ok)
    (z3, _), _ = encode_day(net, data3, lay, p, t)
    assert not torch.equal(z[:, 0], z3[:, 0])


def test_policy_weights_are_long_only_capped_and_masked() -> None:
    net = small_model()
    torch.manual_seed(1)
    b, s, d = 3, 7, net.cfg.d_model
    valid = torch.rand(b, s) > 0.3
    valid[2] = False  # a day with no company: all cash
    w = net.policy(torch.randn(b, s, d), torch.randn(b, s, 3, 2) * 0.01, valid, torch.zeros(b, s))
    assert w.shape == (b, s + 1)
    assert torch.all(w >= 0) and torch.allclose(w.sum(1), torch.ones(b))
    assert torch.all(w[:, :-1] <= net.cfg.max_weight + 1e-7)
    assert torch.all(w[:, :-1][~valid] == 0) and w[2, -1] == 1.0


def test_objective_helpers() -> None:
    w = torch.tensor([[0.5, 0.25]])
    r = torch.tensor([[math.log(2.0), 0.0]])
    # 0.5 doubles to 1.0; total 1.0 + 0.25 + 0.25 cash = 1.5.
    torch.testing.assert_close(drift(w, r), torch.tensor([[1.0 / 1.5, 0.25 / 1.5]]))
    x = torch.arange(100.0)
    assert cvar(x, 0.95).item() == pytest.approx(97.0)  # mean of 95..99
    y, mu, ls = torch.tensor([0.3, -1.0]), torch.tensor([0.1, 0.2]), torch.tensor([-1.0, 0.5])
    ref = -torch.distributions.StudentT(5.0, mu, ls.exp()).log_prob(y)
    torch.testing.assert_close(student_t_nll(y, mu, ls, 5.0), ref)


def strategy(g, **train_kw) -> GraphTransformerStrategy:  # type: ignore[no-untyped-def]
    kw = dict(
        epochs=2,
        warm_epochs=1,
        steps_per_epoch=3,
        val_days=30,
        window_days=10,
        burn_in=2,
        batch_windows=2,
        universe=6,
        cost_bps=5.0,
    )
    kw.update(train_kw)
    return GraphTransformerStrategy(
        g,
        model=dict(window=16, patch=4, d_model=16, heads=2, graph_layers=1, horizons=(1, 5)),
        train=TrainConfig(**kw),  # type: ignore[arg-type]
        overlay=Overlay(),
    )


def test_objective_is_differentiable_end_to_end() -> None:
    p, g, _ = universe(stagger=0)
    st = strategy(g, epochs=1, steps_per_epoch=1, universe=None)
    st.fit(p.truncate(200), np.arange(0, 170))
    assert st.net is not None
    data = PanelData.build(p.truncate(200), (1, 5))
    lay = st.layout(p)
    days = (torch.tensor([40, 90])[:, None] + torch.arange(10)).flatten()
    z, preds, valid = st._encode(data, lay, p.decision_time, days)
    g_ = st._rollout(
        z.view(2, 10, *z.shape[1:]),
        preds.view(2, 10, *preds.shape[1:]),
        valid.view(2, 10, -1),
        data.intraday[days].view(2, 10, -1),
        data.r1[days].view(2, 10, -1),
    )
    st._objective(g_).backward()
    grads = [p_.grad for p_ in st.net.score.parameters()]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    assert any(gr.abs().sum() > 0 for gr in grads if gr is not None)
    assert torch.isfinite(g_).all()


def test_strategy_is_causal_deterministic_and_warm_starts() -> None:
    p, g, _ = universe()
    st = strategy(g)
    st.fit(p.truncate(200), np.arange(0, 170))
    assert st.history and all(math.isfinite(h["val_objective"]) for h in st.history)
    steps = np.arange(200, 230)
    w = st.targets(p, steps)
    assert w.shape == (steps.size, S)
    held = np.nan_to_num(w)
    assert (held >= 0).all() and (held.sum(1) <= 1 + 1e-9).all()
    check_causal(st, p, steps, probes=[0, 7, 29], full=w)
    np.testing.assert_array_equal(st.targets(p, steps), w)
    # The next fold continues from the carried book, and new classes extend
    # the embedding table.
    vocab = len(st.vocab)
    st.fit(p.truncate(230), np.arange(0, 200))
    assert len(st.vocab) >= vocab
    w2 = st.targets(p, np.arange(230, 250))
    assert np.isfinite(np.nan_to_num(w2)).all()
    with pytest.raises(ValueError):
        TrainConfig(burn_in=30, window_days=24)
    with pytest.raises(ValueError):
        Overlay(start=0.3, full=0.2)
