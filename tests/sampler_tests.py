import pytest

from reinforce.sampler import DataSampler

@pytest.fixture
def sampler():
    return DataSampler("../data/example.db")

def test_sampler(sampler: DataSampler):
    c = sampler.counter

    assert c == 0, "Sampler counter did not start at 0"

    row = sampler._fetch_row(0)

    assert len(row) > 0, "Sampler returned empty row"

    done, close, state = sampler.sample_next()

    assert done == False, "Sampler returned done prematurely"
    assert sampler.counter == 1, "Sampler counter didn't increase"
    assert close > 0, "Sampler returned improper close value"
    assert len(state) == len(sampler._queue), "Sampler returned the wrong sized state"

    ct = sampler._rows-2
    sampler._counter = ct
    done, close, state = sampler.sample_next()

    assert done == True, "Sampler did not return done correctly"
    assert sampler.counter == ct, "Sampler counter increased"
    assert close > 0, "Sampler returned improper close value"
    assert len(state) == len(sampler._queue), "Sampler returned the wrong sized state"

    sampler.reset()

    assert sampler.counter == 0, "Sampler counter didn't reset"
