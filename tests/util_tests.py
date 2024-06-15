import pytest
import numpy as np

from reinforce.utils import *

data_rows = 64
data_cols = 7
close_col = 4
high_col  = 2
low_col   = 3

@pytest.fixture
def data():
    return np.random.rand(data_rows, data_cols)

def test_sharpe():
    returns = [1.]
    sharpe = compute_sharpe_ratio(returns, 1.0)

    assert sharpe == 0, "Sharpe ratio should have returned 0"

    returns = [1.0, 1.5]
    sharpe = compute_sharpe_ratio(returns, 0.5)

    assert sharpe > 0, "Sharpe ratio returned negative value"

def test_sma(data):
    close = data[:, close_col]
    sma = compute_sma(close)
    assert len(sma) == data_rows, "SMA array returned with the wrong size"

    sma = compute_sma(close, data_rows+1)
    assert sma is None, "SMA should have returned None"

def test_ema(data):
    close = data[:, close_col]
    ema = compute_ema(close)
    assert len(ema) == data_rows, "EMA array returned with the wrong size"

    ema = compute_ema(close, data_rows+1)
    assert ema is None, "EMA should have returned None"

def test_rsi(data):
    close = data[:, close_col]
    rsi = compute_rsi(close)
    assert len(rsi) == data_rows, "RSI array returned with the wrong size"

    rsi = compute_rsi(close, data_rows+1)
    assert rsi is None, "RSI should have returned None"

def test_stochastic(data):
    close, high, low = data[:, close_col], data[:, high_col], data[:, low_col]
    k, d = compute_stochastic_np(close, high, low)
    assert len(k) == data_rows, "Stochastic K array returned with the wrong size"
    assert len(d) == data_rows, "Stochastic D array returned with the wrong size"

    k, d = compute_stochastic_np(close, high, low, window=data_rows+1)
    assert k is None, "Stochastic K should have returned None"
    assert d is None, "Stochastic D should have returned None"

    k, d = compute_stochastic_np(close, high[1:], low[2:])
    assert k is None, "Stochastic K should have returned None"
    assert d is None, "Stochastic D should have returned None"
