import ccxt
import redis
from typing import Union

from legacy.simulator import Simulator

class Wallet:
    def __init__(
            self, 
            exchange: str, 
            interval: str = "1m", 
            asset: str = "BTC/USDT",
            funds: float = 10_000.0,
            buffer_size: int = 720,
            host: str = "localhost",
            port: int = 6379,
        ) -> None:

        self.buffer_size = buffer_size
        self.asset = asset
        self.interval = interval

        exchange_mapping = {
            "binanceus": ccxt.binanceus,
            "kraken": ccxt.kraken,
            "coinbasepro": ccxt.coinbasepro,
            "simulator": Simulator
        }

        self.redis_conn = redis.Redis(
            host=host,
            port=port,
            decode_responses=True
        )
        self.init_redis(exchange, interval, asset, funds)

        # self.exchange = ccxt.binanceus()
        if exchange.lower() in exchange_mapping: self.exchange = exchange_mapping[exchange]()
        else: raise NotImplementedError("Modify the Wallet __init__ method to use the exchange you want.")

    def init_redis(self, exchange: str, interval: str, asset: str, funds: float) -> None:
        pipe = self.redis_conn.pipeline()
        pipe.set("exchange", exchange)
        pipe.set("interval", interval)
        pipe.set("asset_class", asset)
        pipe.set("holdings", 0)
        pipe.set("cash", funds)
        pipe.execute()

    def fetch_holdings(self) -> float:
        holdings = self.redis_conn.get("holdings")
        return float(holdings)

    def fetch_cash(self) -> float:
        cash = self.redis_conn.get("cash")
        return float(cash)
    
    def set_holdings(self, amount: float) -> None:
        self.redis_conn.set("holdings", amount)

    def set_cash(self, amount: float) -> None:
        self.redis_conn.set("cash", amount)

    def get_fee_rate(self) -> float:
        # TODO Implement it...
        return 0.01

    def last_traded_price(self, asset: str) -> Union[float, None]:
        try:
            ticker = self.exchange.fetch_ticker(asset)
            return ticker['last']
        except Exception as e:
            print(f"Error fetching last traded price: {e}")
            return None

    def mid_market_price(self, asset: str) -> Union[float, None]:
        try:
            order_book = self.exchange.fetch_order_book(asset)
            best_bid = order_book['bids'][0][0] if order_book['bids'] else 0
            best_ask = order_book['asks'][0][0] if order_book['asks'] else 0
            return (best_bid + best_ask) / 2 if best_bid and best_ask else None
        except Exception as e:
            print(f"Error fetching order book: {e}")
            return None

    def depth_weighted_price(self, asset, depth=5) -> Union[float, None]:
        try:
            order_book = self.exchange.fetch_order_book(asset)
            total_volume = sum([bid[1] for bid in order_book['bids'][:depth]]) + sum([ask[1] for ask in order_book['asks'][:depth]])
            weighted_price = sum([bid[0] * bid[1] for bid in order_book['bids'][:depth]]) + sum([ask[0] * ask[1] for ask in order_book['asks'][:depth]])
            return weighted_price / total_volume if total_volume else None
        except Exception as e:
            print(f"Error fetching order book depth: {e}")
            return None

    def net_worth(self, pricing_method: str='last_traded_price') -> Union[float, None]:
        try:
            asset_price = None

            if pricing_method == 'last_traded_price':
                asset_price = self.last_traded_price(self.asset)
            elif pricing_method == 'mid_market_price':
                asset_price = self.mid_market_price(self.asset)
            elif pricing_method == 'depth_weighted_price':
                asset_price = self.depth_weighted_price(self.asset)
            else:
                raise ValueError(f"Invalid pricing method: {pricing_method}")

            if asset_price is None:
                raise ValueError("Unable to fetch asset price")

            holdings_value = self.fetch_holdings() * asset_price
            cash = self.fetch_cash()

            return holdings_value + cash
        except Exception as e:
            print(f"Error calculating net worth with {pricing_method}: {e}")
            return None

    def buy(self, asset: str, amount: Union[float, str]) -> bool:
        """
        Simulates buying an amount of a specified asset.
        :param asset: The asset to buy (e.g., 'BTC/USD').
        :param amount: The amount of the asset to buy, or 'max' to use all available cash.
        :return: True if the buy was successful, False otherwise.
        """
        try:
            price = self.last_traded_price(asset)
            if price is None or price == 0:
                raise ValueError("Unable to fetch a valid asset price")

            fee_rate = self.get_fee_rate()  # Fetch the current fee rate
            cash = self.fetch_cash()

            # Calculate the maximum purchasable amount if 'max' is specified
            if amount == "max":
                total_cost_per_unit = price * (1 + fee_rate)
                amount = cash / total_cost_per_unit

            total_cost = amount * price
            total_cost_incl_fee = total_cost * (1 + fee_rate)

            if cash < total_cost_incl_fee:
                raise ValueError(f"Insufficient funds, available: {cash:.2f}, cost: {total_cost_incl_fee:.2f}")

            # Update cash and holdings
            self.set_cash(cash - total_cost_incl_fee)
            self.set_holdings(self.fetch_holdings() + amount)

            return True
        except Exception as e:
            print(f"Error in buying: {e}")
            return False

    def sell(self, asset: str, amount: float) -> bool:
        """
        Simulates selling an amount of a specified asset.
        :param asset: The asset to sell (e.g., 'BTC/USD').
        :param amount: The amount of the asset to sell.
        :return: True if the sell was successful, False otherwise.
        """
        try:
            price = self.last_traded_price(asset)
            if price is None:
                raise ValueError("Unable to fetch asset price")

            fee_rate = self.get_fee_rate()  # Fetch the current fee rate
            total_revenue = amount * price
            total_revenue_after_fee = total_revenue * (1 - fee_rate)

            holdings = self.fetch_holdings()
            if holdings < amount:
                raise ValueError("Insufficient holdings")

            # Update cash and holdings
            self.set_cash(self.fetch_cash() + total_revenue_after_fee)
            self.set_holdings(holdings - amount)
            
            return True
        except Exception as e:
            print(f"Error in selling: {e}")
            return False

    def fill_one(self, asset: str) -> None:
        try:
            # Fetching the OHLCVC data
            ohlcvc_data = self.exchange.fetch_ohlcv(asset, self.interval, limit=2)
            if not ohlcvc_data:
                raise ValueError(f"No data received for {asset}")

            latest_data = ohlcvc_data[-1]
            if len(latest_data) < 6:
                raise ValueError(f"Incomplete data for {asset}: {latest_data}")

            # Fetch the last timestamp from Redis
            last_timestamp = self.redis_conn.lindex('timestamp', -1)

            # Check if the new data's timestamp is different from the last timestamp
            if last_timestamp is None or last_timestamp != latest_data[0]:
                pipe = self.redis_conn.pipeline()
                order = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] #, 'count']
                for key, value in zip(order, latest_data):
                    # Append value to the end of the list
                    pipe.rpush(key, value)

                    # Trim the list to maintain the maximum length
                    if self.buffer_size is not None:
                        pipe.ltrim(key, -self.buffer_size, -1)

                pipe.execute()
            else:
                print(f"Duplicate timestamp for {asset}, skipping data insertion.")

        except Exception as e:
            print(f"Error in fill_one method for {asset}: {e}")

    def fill_buffer(self, asset: str) -> None:
        try:
            # Fetching the OHLCVC data
            ohlcvc_data = self.exchange.fetch_ohlcv(asset, self.interval, limit=self.buffer_size)
            if not ohlcvc_data:
                raise ValueError(f"No data received for {asset}")

            # Transposing the OHLCVC data
            transposed_data = list(map(list, zip(*ohlcvc_data)))

            # Using Redis pipeline for efficient data storage
            pipe = self.redis_conn.pipeline()
            order = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] #, 'count']

            # Ensure we have the expected number of components
            if len(transposed_data) != len(order):
                raise ValueError(f"Unexpected data structure for {asset}")

            # Push each component to its corresponding Redis list
            for key, values in zip(order, transposed_data):
                pipe.rpush(key, *values)

            pipe.execute()

        except Exception as e:
            print(f"Error in fill_buffer method for {asset}: {e}")