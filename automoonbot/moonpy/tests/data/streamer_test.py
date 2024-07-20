import time
import pytest
from automoonbot.moonpy.data import Streamer

DATA_SIZE = 10
QUEUE_SIZE = 1

@pytest.fixture
def data():
    return [1] * DATA_SIZE

def test_basics(data):
    streamer = Streamer(QUEUE_SIZE, data=data, sleep=1)
    assert not streamer.running, "streamer shouldn't be running"
    streamer.start()
    assert streamer.running, "streamer should be running"
    val = next(streamer)
    assert val == 1, "queue should have data"
    time.sleep(0.5)
    val = next(streamer)
    assert val is None, "queue should not have data"
    time.sleep(0.6)
    val = next(streamer)
    assert val == 1, "queue should have data"
    streamer.stop()
    assert not streamer.running, "streamer shouldn't be running"

def test_prefill(data):
    streamer = Streamer(QUEUE_SIZE, data=data, sleep=1)
    streamer.prefill()
    val = next(streamer)
    assert val == 1, "queue should have data from prefill"

def test_thread_safe_start_stop(data):
    streamer = Streamer(QUEUE_SIZE, data=data, sleep=1)
    streamer.start()
    assert streamer.running, "streamer should be running"
    streamer.stop()
    assert not streamer.running, "streamer shouldn't be running"
    streamer.start()
    assert streamer.running, "streamer should be running again"
    streamer.stop()
    assert not streamer.running, "streamer shouldn't be running again"

def test_iterator_protocol(data):
    streamer = Streamer(QUEUE_SIZE, data=data, done="raise", sleep=1)
    streamer.start()
    val = next(streamer)
    assert val == 1, "queue should have data"
    with pytest.raises(StopIteration):
        next(streamer)
    streamer.stop()