import math
import cbpro
import datetime
import pandas as pd
from cbpro.public_client import PublicClient as pcl
from authCredentials import api_secret, api_key, api_pass

client = cbpro.AuthenticatedClient(
    api_key,
    api_secret,
    api_pass,
    api_url = 'https://api.pro.coinbase.com')

def round_decimals_down(number:float, decimals:int=2):
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)
    factor = 10 ** decimals
    return math.floor(number * factor) / factor
    
def rightNow():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
def currentPrice(asset):
    req = pcl()
    return float(req.get_product_ticker(asset)['price'])
    
def balance(asset):
    acc = pd.DataFrame(client.get_accounts())
    if asset == 'USD-USD':
        return float(acc.loc[acc['currency'] == 'USD', 'balance'])
    else:
        return round_decimals_down(float(acc.loc[acc['currency'] == asset.split('-')[0], 'balance']), 8)
    
def buy(asset, amount):
    if amount > 0 and amount <= balance('USD-USD'):
        client.place_market_order(product_id = asset, side = 'buy', funds = round_decimals_down(amount, 7))
        print(f"Bought {asset.split('-')[0]} at ${currentPrice(asset)}, datetime: {rightNow()} \n")
    else:
        print("Insufficient Funds \n")

def sell(asset):
    if balance(asset) > 0:
        client.place_market_order(product_id = asset, side = 'sell', size = balance(asset))
        print(f"Sold all {asset.split('-')[0]} at ${currentPrice(asset)}, datetime: {rightNow()} \n")
    else:
        print("Insufficient Asset \n")

