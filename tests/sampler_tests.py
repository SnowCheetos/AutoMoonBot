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

