from reinforce.environment import TradeEnv, train
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    env = TradeEnv(
        state_dim=40,
        action_dim=3,
        embedding_dim=4,
        queue_size=70,
        inaction_cost=-5,
        action_cost=0,
        device="cpu",
        db_path="data/example.db",
        return_thresh=0.9
    )

    history = train(env, 1000, 1e-3)

    print(history)