import time
import pytest
from backend.data import Streamer

DATA_SIZE = 10
QUEUE_SIZE = 1

@pytest.fixture
def data():
    return [1] * DATA_SIZE

def test_basics(data):
    streamer = Streamer(QUEUE_SIZE, data, sleep=1)
    assert not streamer.running, "streamer shouldn't be running"

    streamer.start()
    assert streamer.running, "streamer should be running"

    val = next(streamer)
    assert val, "queue should have data"

    time.sleep(0.5)
    val = next(streamer)
    assert not val, "queue should not have data"

    time.sleep(0.6)
    val = next(streamer)
    assert val, "queue should have data"

    streamer.stop()
    assert not streamer.running, "streamer shouldn't be running"
