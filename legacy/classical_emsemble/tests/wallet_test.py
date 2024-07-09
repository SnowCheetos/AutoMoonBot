import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.wallet import Wallet

class TestWallet(unittest.TestCase):

    def setUp(self):
        self.wallet = Wallet(exchange='binanceus', host='localhost', port=6379)

    def tearDown(self):
        # Flush Redis after each test
        self.wallet.redis_conn.flushall()

    def test_fetch_and_set_cash(self):
        # Set a cash value
        initial_cash = 1000.0
        self.wallet.set_cash(initial_cash)

        # Fetch the cash value
        cash = self.wallet.fetch_cash()
        self.assertEqual(cash, initial_cash)

    def test_fetch_and_set_holdings(self):
        # Set a holdings value
        initial_holdings = 1.5  # Assuming 1.5 BTC for example
        self.wallet.set_holdings(initial_holdings)

        # Fetch the holdings value
        holdings = self.wallet.fetch_holdings()
        self.assertEqual(holdings, initial_holdings)

    def test_last_traded_price(self):
        # Fetch the last traded price
        price = self.wallet.last_traded_price('BTC/USDT')
        self.assertIsNotNone(price)
        self.assertIsInstance(price, float)

    def test_buy(self):
        # Assuming you have sufficient initial cash for testing
        initial_cash = 10000.0
        buy_amount = 0.01  # Small amount for testing
        self.wallet.set_cash(initial_cash)

        # Fetch the current price for a small, controlled buy operation
        current_price = self.wallet.last_traded_price('BTC/USDT')
        self.assertIsNotNone(current_price)

        # Perform the buy operation
        success = self.wallet.buy('BTC/USDT', buy_amount)
        self.assertTrue(success)

        # Check if cash is reduced and holdings are increased appropriately
        expected_cash = initial_cash - (buy_amount * current_price) * (1 + self.wallet.get_fee_rate())
        self.assertAlmostEqual(self.wallet.fetch_cash(), expected_cash, places=2)

        # Assuming no initial holdings
        self.assertAlmostEqual(self.wallet.fetch_holdings(), buy_amount, places=8)

    def test_sell(self):
        # Set initial holdings and cash
        initial_holdings = 0.01  # Small amount for testing
        initial_cash = 1000.0
        self.wallet.set_holdings(initial_holdings)
        self.wallet.set_cash(initial_cash)

        # Fetch the current price for a small, controlled sell operation
        current_price = self.wallet.last_traded_price('BTC/USDT')
        self.assertIsNotNone(current_price)

        # Perform the sell operation
        success = self.wallet.sell('BTC/USDT', initial_holdings)
        self.assertTrue(success)

        # Check if cash is increased and holdings are reduced appropriately
        expected_cash = initial_cash + (initial_holdings * current_price) * (1 - self.wallet.get_fee_rate())
        self.assertAlmostEqual(self.wallet.fetch_cash(), expected_cash, places=2)

        # Holdings should be zero after selling all
        self.assertEqual(self.wallet.fetch_holdings(), 0)

if __name__ == '__main__':
    unittest.main()