import pytest

from reinforce.environment import TradeEnv, train

@pytest.fixture
def env():
    return TradeEnv(
        state_dim=40,
        action_dim=3,
        embedding_dim=4,
        queue_size=70,
        inaction_cost=-5,
        action_cost=0,
        device="cpu",
        db_path="../data/example.db",
        return_thresh=0.9
    )

def test_env(env):
    history = train(env, 1)
    assert len(history) == 1, "bad"
