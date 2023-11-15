import time
import logging
import numpy as np
from src.wallet import Wallet
from src.trainer import Trainer

class Trader(Wallet):
    def __init__(
            self, 
            exchange: str, 
            interval: str = "1m", 
            asset: str = "BTC/USDT", 
            funds: float = 10_000.0, 
            buffer_size: int = 100, 
            host: str = "localhost", 
            port: int = 6379,
            risk: float = 0.05, 
            multiplier: float = 1.025, 
            cycle_time: int = 60 * 30,
            max_cycles: int = 1000
        ) -> None:
        super().__init__(exchange, interval, asset, funds, buffer_size, host, port)

        self.risk = risk
        self.multiplier = multiplier
        self.cycle_time = cycle_time
        self.max_cycles = max_cycles

        self.init_trade_params()

    def init_trade_params(self) -> None:
        self.status = "idle"
        pipe = self.redis_conn.pipeline()
        pipe.set("stop_loss", 0.0)
        pipe.set("take_profit", "inf")
        pipe.execute()

    @property
    def stop_loss(self) -> float:
        return float(self.redis_conn.get("stop_loss"))

    @stop_loss.setter
    def stop_loss(self, val: float) -> None:
        self.redis_conn.set("stop_loss", val)

    @property
    def take_profit(self) -> float:
        return float(self.redis_conn.get("take_profit"))

    @take_profit.setter
    def take_profit(self, val: float) -> None:
        self.redis_conn.set("take_profit", val)

    @property
    def status(self) -> str:
        return self.redis_conn.get("status")
    
    @status.setter
    def status(self, val: str) -> None:
        return self.redis_conn.set("status", val)
    
    @property
    def prediction(self) -> str:
        return self.redis_conn.get("prediction")
    
    @prediction.setter
    def prediction(self, val: str) -> None:
        return self.redis_conn.set("prediction", val)

    def reset_timer(self) -> None:
        self.redis_conn.setex("timer", self.cycle_time, 1)

    def fetch_timer(self) -> float:
        return float(self.redis_conn.ttl("timer"))

    def predict(self):
        # Sends message to message queue, model service will predict and store result in redis
        pass

    def try_trade(self):
        price = self.depth_weighted_price(self.asset)
        assert price is not None, "Error fetching price"

        prediction = self.prediction
        if self.status == "idle":
            if prediction == "buy":
                self.status = "buy_spec"
                self.take_profit = price * (1 - self.multiplier * self.risk)
                self.stop_loss = price * (1 + self.risk)

            elif prediction == "sell":
                self.state = "sell_spec"
                self.take_profit = price * (1 + self.risk)
                self.stop_loss = price * (1 - self.multiplier * self.risk)

        elif (self.status == "buy_spec") and (prediction == "buy") and (price >= self.stop_loss or price <= self.take_profit):
            s = self.buy(self.asset, self.max_buy(self.asset))
            self.init_trade_params()

        elif (self.status == "sell_spec") and (prediction == "sell") and (price <= self.stop_loss or price >= self.take_profit):
            s = self.sell(self.asset, self.fetch_holdings())
            self.init_trade_params()

    def run(self):
        for cycle in range(self.max_cycles):
            self.reset_timer()

            clock = self.fetch_timer()
            while clock > 0:
                self.fill_one(self.asset)
                self.predict()

                time.sleep(5)
                self.try_trade()

                time.sleep(60 - (clock - self.fetch_timer()))