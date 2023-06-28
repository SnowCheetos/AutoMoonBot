import ccxt
import numpy as np
import pandas as pd
from collections import deque

class Wallet:
    def __init__(self, exchange, configs={}, timestep="1m", asset="BTC/USDT", max_buffer_size=720, simulation=True, funds=10_000, sim_file="data/data.csv"):
        self.simulation = simulation
        self.funds = funds
        self.assets = 0
        self.timestep = timestep

        self.asset = asset
        exchange_mapping = {
            "BinanceUS": ccxt.binanceus,
            "Kraken": ccxt.kraken,
            "CoinbasePro": ccxt.coinbasepro,
        }
        
        if exchange in exchange_mapping:
            self.exchange = exchange_mapping[exchange](configs)
        else:
            raise NotImplementedError("Modify the Wallet __init__ method to use the exchange you want.")

        self.max_buffer = max_buffer_size
        self.buffer = deque(maxlen=max_buffer_size)

        self.sim_file = sim_file
        self.data = None
        self.data_counter = max_buffer_size
        if self.sim_file:
            self.data = pd.read_csv(sim_file)

        self.fill_buffer(self.asset)
        self.action = None

    def net_worth(self, price):
        usd = self.funds
        ass = self.assets * price
        return ass + usd

    def check_balance(self, asset):
        if self.simulation:
            if asset == "USD" or asset == "USDT" or asset == "USDC":
                return self.funds
            return self.assets
        raise NotImplementedError("Modify the Wallet check_balance method.")

    def execute_buy(self, asset, amount, sim_price=None):
        print("[INFO] Executing buy...")
        price = self.market_price(asset, "asks")
        if self.simulation:
            price = sim_price
            if amount == "all":
                amount = (self.funds / price) * 0.9999
            if amount * price < self.funds:
                self.assets += amount
                self.funds -= amount * price
                self.action = -1
            else:
                print("Not enough funds.")
        else:
            raise NotImplementedError("Modify the Wallet execute_buy method.")

    def execute_sell(self, asset, amount, sim_price=None):
        print("[INFO] Executing sell...")
        price = self.market_price(asset, "bids")
        if self.simulation:
            price = sim_price
            if amount == "all":
                amount = 0.9999*self.assets
            if amount <= self.assets:
                self.assets -= amount
                self.funds += amount * price
                self.action = 1
            print("Not enough assets")
        else:
            raise NotImplementedError("Modify the Wallet execute_sell method.")

    def fetch_one(self, asset):
        if self.simulation:
            data = self.data.iloc[self.data_counter].to_numpy()[1:]
            data = np.hstack([[1], data]).astype(np.float32)
            self.buffer.append(data)
            self.data_counter += 1
        else:
            data = self.exchange.fetch_ohlcv(asset, self.timestep, limit=1)
            self.buffer.append(data[0])
    
    def fill_buffer(self, asset):
        if self.simulation:
            data = self.data[:self.max_buffer].to_numpy()[:, 1:]
            data = np.hstack([np.ones((data.shape[0], 1)), data]).astype(np.float32)
        else:
            data = self.exchange.fetch_ohlcv(asset, self.timestep, limit=self.max_buffer)
        for d in data:
            self.buffer.append(d)
    
    def market_price(self, asset, side):
        data = self.exchange.fetch_order_book(asset)
        if side == "bids":
            return data.get(side)[0][0]
        elif side == "asks":
            return data.get(side)[0][0]
        else:
            return 0.5 * (data.get("bids")[0][0] + data.get("asks")[0][0])
    
    def fetch(self):
        return np.asarray(self.buffer)[:, 1:]