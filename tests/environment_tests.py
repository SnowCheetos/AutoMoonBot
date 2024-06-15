import pytest

from reinforce.environment import TradeEnv, train

@pytest.fixture
def env():
    return TradeEnv(
        state_dim=40,
        action_dim=3,
        embedding_dim=8,
        queue_size=70,
        device="cpu",
        db_path="../data/example.db",
        return_thresh=0.5
    )

def test_env(env):
    history = train(env, 1)

    assert len(history) == 1, "bad"