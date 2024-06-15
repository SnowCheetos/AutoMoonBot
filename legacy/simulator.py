import pandas as pd

class Simulator:
    def __init__(self, data: str = "data/data.csv") -> None:
        self.i = 720
        self.data = pd.read_csv(data).to_numpy()

    def fetch_ohlcv(self, asset, _, limit):
        ohlcvc = self.data[max(self.i - limit, 0):self.i]
        self.i = min(self.i + 1, len(self.data))
        return [list(row) for row in ohlcvc]

    def fetch_ticker(self, asset):
        ticker_data = {"last": self.data[min(self.i, len(self.data) - 1), 4]}
        self.i = min(self.i + 1, len(self.data))
        return ticker_data

    def fetch_order_book(self, asset):
        i = min(self.i, len(self.data) - 1)
        bid = self.data[i-7:i, 3]
        ask = self.data[i-7:i, 2]
        order_book = {"bids": [bid], "asks": [ask]}
        self.i = min(self.i + 1, len(self.data))
        return order_book
