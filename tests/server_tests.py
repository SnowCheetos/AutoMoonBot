import time
import pytest
import numpy as np

from backend.server import Server


@pytest.fixture
def server():
    return Server(
        ticker="SPY",
        period="3mo",
        interval="1h",
        queue_size=70,
        state_dim=40,
        action_dim=3,
        embedding_dim=8,
        inaction_cost=-1,
        action_cost=0,
        device="cpu",
        db_path="../data/example.db",
        return_thresh=0.9,
        feature_params={
            "sma": np.geomspace(8, 64, 8).astype(int).tolist(),
            "ema": np.geomspace(8, 64, 8).astype(int).tolist(),
            "rsi": np.geomspace(4, 64, 8).astype(int).tolist(),
            "sto": {
                "window": np.geomspace(8, 64, 4).astype(int).tolist() + np.geomspace(8, 64, 4).astype(int).tolist(),
                "k":      np.linspace(3, 11, 8).astype(int).tolist(),
                "d":      np.linspace(3, 11, 8).astype(int).tolist()
            }
        }
    )

def test_inference(server: Server):
    a = server.run_inference(True)
    assert a is not None, "Inference returned None"

def test_training(server: Server):
    start = time.time()
    server.train_model(1)
    stop = time.time()

    assert server.busy, "Server should be busy"
    assert stop - start < 1, "Main thread should be free"

    time.sleep(10) # kinda wack...
    assert not server.busy, "Server should be free"

def test_pipeline(server: Server):
    server.train_model(10)

    a = server.run_inference(True)
    assert a is not None, "Inference returned None"
    assert server.busy, "Server should be busy"

    m = server.update_model()

    assert not m, "Server should not have copied weights"

    t = server.join_train_thread()

    assert t, "Server did not properly join training thread"
    assert not server.busy, "Server should be free"

    m = server.update_model()

    assert m, "Server should have copied weights"