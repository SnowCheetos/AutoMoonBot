import pytest
import numpy as np

from reinforce.sampler import DataSampler

@pytest.fixture
def sampler():
    return DataSampler("../data/example.db", 65, {
                "sma": np.geomspace(8, 64, 8, dtype=int).tolist(),
                "ema": np.geomspace(8, 64, 8, dtype=int).tolist(),
                "rsi": np.geomspace(4, 64, 8, dtype=int).tolist(),
                "sto": {
                    "window": np.geomspace(8, 64, 4, dtype=int).tolist() + np.geomspace(8, 64, 4, dtype=int).tolist(),
                    "k": np.linspace(3, 11, 8, dtype=int).tolist(),
                    "d": np.linspace(3, 11, 8, dtype=int).tolist()
                }})

def test_sampler(sampler: DataSampler):
    c = sampler.counter

    assert c == 0, "Sampler counter did not start at 0"

    row = sampler._fetch_row(0)

    assert len(row) > 0, "Sampler returned empty row"

    done, close, state = sampler.sample_next()

    assert done == False, "Sampler returned done prematurely"
    assert sampler.counter == 1, "Sampler counter didn't increase"
    assert close > 0, "Sampler returned improper close value"
    assert len(state) == 0, "Sampler should have returned None state"

    ct = sampler._rows-2
    sampler._counter = ct
    done, close, state = sampler.sample_next()

    assert done == True, "Sampler did not return done correctly"
    assert sampler.counter == ct, "Sampler counter increased"
    assert close > 0, "Sampler returned improper close value"
    assert len(state) == 0, "Sampler should have returned None state"

    sampler.reset()

    assert sampler.counter == 0, "Sampler counter didn't reset"

    done, close, state = sampler.sample_next()
    while len(state) == 0:
        done, close, state = sampler.sample_next()

    assert len(state) == 1, "Sampler should have returned valid state vector"