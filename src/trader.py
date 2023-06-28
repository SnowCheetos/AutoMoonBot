import time
import logging
import numpy as np
from src.wallet import Wallet
from src.trainer import Trainer

class Trader(Wallet):
    def __init__(self, asset, model: Trainer, risk=0.05, multiplier=1.025, timestep="1m", exchange="BinanceUS", configs={}):
        super().__init__(
            exchange, 
            configs=configs, 
            timestep=timestep, 
            asset=asset, 
            max_buffer_size= 300, 
            simulation= True, 
            funds= 10000, 
            sim_file="data/data.csv")

        self.DELAY = 60
        self.state = 0
        self._take_profit = np.nan
        self._stop_loss = np.nan
        self.risk = risk
        self.multiplier = multiplier
        self.model = model

        logging.basicConfig(filename='trader.log', level=logging.INFO)

    @property
    def take_profit(self):
        return self._take_profit

    @take_profit.setter
    def take_profit(self, value):
        self._take_profit = value
        logging.info(f'Set take profit to {value}')

    @property
    def stop_loss(self):
        return self._stop_loss

    @stop_loss.setter
    def stop_loss(self, value):
        self._stop_loss = value
        logging.info(f'Set stop loss to {value}')

    def update_state(self, new_state):
        self.state = new_state
        logging.info(f'Set state to {new_state}')

    def on_new_data(self, data):
        try:
            X = np.array(data)[:,:-1]
            price = np.array(data['Close'])
            prediction = self.model.predict(X)
            self.tryTrade(prediction[-1], price)
        except Exception as e:
            logging.error(f'Error occurred during trading: {str(e)}')

    def try_trade(self, prediction, price):
        self.action = None
        print("[INFO] Trying to trade... \n")
        print(f"[INFO] Price: {price} | Stop Loss: {self.stop_loss} | Take Profit: {self.take_profit} | State: {self.state}\n")
        if self.state == 0:  # No current position, so can make a new trade decision
            if prediction == -1:  # Can buy only if we have USD
                self.state = prediction
                self.take_profit = price * (1 - self.multiplier * self.risk)
                self.stop_loss = price * (1 + self.risk)
                print("[INFO] New trading decision made for buying. \n")
            elif prediction == 1:  # Can sell only if we have the asset
                self.state = prediction
                self.take_profit = price * (1 + self.risk)
                self.stop_loss = price * (1 - self.multiplier * self.risk)
                print("[INFO] New trading decision made for selling. \n")
        elif self.state != 0:  # Already in a position, check if profit or stop loss hit
            if (prediction == -1 and (price > self.stop_loss or price < self.take_profit)) or \
            (prediction == 1 and (price < self.stop_loss or price > self.take_profit)):
                operation = self.execute_buy if prediction == -1 else self.execute_sell
                operation(self.asset, "all", sim_price=price)
                self.state = 0  # reset state after execution
                self.take_profit = np.nan
                self.stop_loss = np.nan
                print("[INFO] Trade executed and position closed. \n")
            else:
                print("[INFO] Position still open, profit or stop loss not hit. \n")
        else:  # state and prediction are not the same, do nothing
            print("[INFO] No action required. \n")
    
    def step(self, train):
        self.fetch_one(self.asset)
        x = self.fetch()
        if train:
            self.model.fit(x)
        if self.simulation:
            self.try_trade(self.model(x)[-1], x[-1, 3])
        else:
            self.try_trade(self.model(x)[-1], self.market_price(self.asset, side=None))
        time.sleep(self.DELAY)
        return self.net_worth(x[-1, 3]), x[-1, 3], self.action
        