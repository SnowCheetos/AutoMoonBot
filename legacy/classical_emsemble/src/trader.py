import json
import time
import pika
import logging

from typing import Union
from src.wallet import Wallet

class Trader(Wallet):
    def __init__(
            self, 
            exchange: str, 
            interval: str = "1m", 
            asset: str = "BTC/USDT", 
            funds: float = 10_000.0, 
            buffer_size: int = 720, 
            host: str = "localhost", 
            port: int = 6379,
            risk: float = 0.05, 
            multiplier: float = 1.025, 
            cycle_time: int = 60,
            max_cycles: int = 1000,
            time_per_pred: Union[int, None] = None
        ) -> None:
        super().__init__(exchange, interval, asset, funds, buffer_size, host, port)

        self.risk = risk
        self.multiplier = multiplier
        self.cycle_time = cycle_time
        self.max_cycles = max_cycles
        interval_map = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
            "3d": 259200,
            "1w": 604800,
            "1M": 2592000,
        }
        self.time_per_pred = interval_map[interval] if not time_per_pred else time_per_pred

        self.init_trade_params()

    @staticmethod
    def send_message(message_body: str, queue='pred_service'):
        # Connect to RabbitMQ
        connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
        channel = connection.channel()
        channel.queue_declare(queue=queue)
        channel.basic_publish(exchange='', routing_key=queue, body=message_body)
        connection.close()

    def init_trade_params(self) -> None:
        self.status = "idle"
        self.prediction = "hold"
        pipe = self.redis_conn.pipeline()
        pipe.set("stop_loss", "nan")
        pipe.set("take_profit", "nan")
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
        data = json.dumps({"ops": "pred"})
        self.send_message(data)

    def train(self):
        data = json.dumps({"ops": "train"})
        self.send_message(data)

    def try_trade(self):
        price = self.depth_weighted_price(self.asset)
        assert price is not None, "Error fetching price"

        prediction = self.prediction

        print(price, prediction, self.status)

        if self.status == "idle":
            if prediction == "buy":
                print("Buy spec...")
                self.status = "buy_spec"
                self.take_profit = price * (1 - self.multiplier * self.risk)
                self.stop_loss = price * (1 + self.risk)

            elif prediction == "sell":
                print("Sell spec...")
                self.status = "sell_spec"
                self.take_profit = price * (1 + self.risk)
                self.stop_loss = price * (1 - self.multiplier * self.risk)
            
            else: print(f"Doing nothing, networth: {self.net_worth()}")

        elif (self.status == "buy_spec") and (price >= self.stop_loss or price <= self.take_profit):
            if self.fetch_cash() > 0:
                s = self.buy(self.asset, "max")
                if s: print(f"Executed buy, networth: {self.net_worth()}")
            else:
                print("No funds to buy, doing nothing...")
            self.init_trade_params()

        elif (self.status == "sell_spec") and (price <= self.stop_loss or price >= self.take_profit):
            if self.fetch_holdings() > 0:
                s = self.sell(self.asset, self.fetch_holdings())
                if s: print(f"Executed sell, networth: {self.net_worth()}")
            else:
                print("No holdings to sell, doing nothing...")
            self.init_trade_params()

        elif (self.status == "buy_spec") and (prediction == "sell"):
            self.init_trade_params()

        elif (self.status == "sell_spec") and (prediction == "buy"):
            self.init_trade_params()

        else: print(f"Doing nothing, networth: {self.net_worth()}")

    def run(self):
        for cycle in range(self.max_cycles):
            print(f"[INFO] Cycle {cycle+1}/{self.max_cycles} Started")
            self.reset_timer()
            self.train()
            clock = self.fetch_timer()
            while clock > 0:
                self.fill_one(self.asset)
                self.predict()

                time.sleep(1)
                self.try_trade()

                sl = clock - self.fetch_timer()
                time.sleep(max(self.time_per_pred - sl, 0))
                clock = self.fetch_timer()
