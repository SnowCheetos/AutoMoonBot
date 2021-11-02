import time
import numpy as np
from subroutines import live_feed
from cbpro_transactions import *
from ensemble import ensembled

class trader:
    def __init__(self, asset):
        self.state = 0
        self.asset = asset
        self.takeProfit = np.nan
        self.stopLoss = np.nan
        self.model = ensembled(asset)

    def run(self, features, cutoff = True):
        while cutoff:
            data = live_feed(self.asset, features)
            X = np.array(data)[:,:-1]
            price = np.array(data['Close'])
            prediction = self.model.predict(X)
            if prediction[-1] == -1:
                self.tryEnter(price)
                time.sleep(900)
                while self.state == -1:
                    self.tryEnter(price)
                    time.sleep(900)
            elif prediction[-1] == 1:
                self.tryExit(price)
                time.sleep(900)
                while self.state == 1:
                    self.tryExit(price)
                    time.sleep(900)
            else:
                print(colored('No actions at ' + rightNow(), 'yellow' , attrs = ['bold']))
            print("""-----------------------------------------------------------------------------------------
            """)
            time.sleep(900)

    def tryEnter(self, price, risk = 0.15, multiplier = 3):
        if self.state == 0:
            self.state = -1
            self.takeProfit = price * (1 - multiplier * risk)
            self.stopLoss = price * (1 + risk)
        elif self.state == -1:
            if price > self.stopLoss or price < self.takeProfit:
                buy(self.asset + '-USD', balance('USD-USD'))
                self.state = 0
                self.takeProfit = np.nan
                self.stopLoss = np.nan
            else:
                self.takeProfit = price * (1 - multiplier * risk)
                self.stopLoss = price * (1 + risk)

    def tryExit(self, price, risk = 0.15, multiplier = 3):
        if self.state == 0:
            self.state = 1
            self.takeProfit = price * (1 + risk)
            self.stopLoss = price * (1 - multiplier * risk) 
        elif self.state == 1:
            if price < self.stopLoss or price > self.takeProfit:
                sell(self.asset + '-USD')
                self.state = 0
                self.takeProfit = np.nan
                self.stopLoss = np.nan
            else:
                self.takeProfit = price * (1 + risk)
                self.stopLoss = price * (1 - multiplier * risk)


if __name__ == '__main__':
    asset = input('Asset: ')
    try:
        features = pd.read_csv('data/features/'+asset+'(200).csv')['feature'].tolist()
    except:
        raise Exception("Data unavailable for given asset. Please generate dataset first.")
    try:
        Trader = trader(asset)
    except:
        raise Exception("Models unavailable for given asset. Please train models first.")
    Trader.run(features)
