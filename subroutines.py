import numpy as np
import pandas as pd
from numba import jit
from numba import int64
from numba import float64
from datetime import datetime
from datetime import timedelta
from fastquant import get_crypto_data

@jit((float64[:], int64), nopython=True, nogil=True)
def EWMA(inp, window):
    n = inp.shape[0]
    ewma = np.empty(n, dtype = np.float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = inp[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1 - alpha)**i
        ewma_old = ewma_old * (1 - alpha) + inp[i]
        ewma[i] = ewma_old / w
    return ewma

def get_data(asset, start, end, interval, source):
    if end == None:
        return get_crypto_data('/'.join([asset, 'USD']),
        (datetime.now() - timedelta(days = start)).strftime("%Y-%m-%d"),
        (datetime.now() + timedelta(days = 1)).strftime("%Y-%m-%d"),
        interval, source)
    return get_crypto_data('/'.join([asset, 'USD']),
    (datetime.now() - timedelta(days = start)).strftime("%Y-%m-%d"),
    (datetime.now() - timedelta(days = end)).strftime("%Y-%m-%d"),
    interval, source)

def live_feed(asset, features, interval = '15m', source = 'coinbasepro'):
    data = get_data(asset, 30, None, interval, source)
    output = pd.DataFrame()
    for feature in features:
        output[feature] = eval(feature)
    output['Close'] = np.array(data['close'])
    return output[int(0.1 * output.shape[0]):]

def labelData(Data, chunkSize):
    data = np.array(Data['close'])
    chunks = [data[x : x + chunkSize] for x in range(0, len(data), chunkSize)]
    B, S = [], []
    buy, sell = [], []
    for chunk in chunks:
        buy.append(chunk[np.argmin(chunk)])
        sell.append(chunk[np.argmax(chunk)])
    for index, price in enumerate(data):
        if price in buy:
            B.append(index)
        elif price in sell:
            S.append(index)
    output = np.zeros(data.shape[0])
    output[B], output[S] = -1, 1
    return output

def ratio(data, column1, column2):
    return np.array(data[column1] / data[column2])

def rGradient(data, column):
    return np.array(np.gradient(data[column]) / data[column])

def RSI(data, column, period):
    up, down = data[column].diff(1).copy(), data[column].diff(1).copy()
    up[up < 0], down[down > 0] = 0, 0
    gain, loss = up.rolling(window = period).mean(), abs(down.rolling(window = period).mean())
    return np.array(0.5 - 1 / (1 + (gain / loss)))

def rEMA(data, column, period):
    EMA = data[column].ewm(span = period).mean()
    return np.array(EMA / data[column])
        
def rSMA(data, column, period):
    SMA = data[column].rolling(window = period).mean()
    return np.array(SMA / data[column])

def rMACD(data, column, long, short, signal):
    L, S = data[column].ewm(span = long, adjust = False).mean(), data[column].ewm(span = short, adjust = False).mean()
    MACD = S - L
    _s_ = MACD.ewm(span = signal, adjust = False).mean()
    return np.array(MACD / _s_)

def rBollinger(data, column, period):
    EMA, STD = data[column].ewm(span = period).mean(), data[column].ewm(span = period).std()
    return np.array([
        (EMA + (2 * STD)) / data[column],
        (EMA - (2 * STD)) / data[column],
        (EMA + (2 * STD)) / (EMA - (2 * STD))])

def rATR(data, period):
    TR = []
    for i in range(data.shape[0]):
        TR.append(
            max(data['high'][i] - data['low'][i],abs(data['high'][i] - data['close'][i]),abs(data['low'][i] - data['close'][i])))
    return np.array([
            EWMA(np.array((data['close'] + 3 * EWMA(np.array(TR), period))), 100) / data['close'],
            EWMA(np.array((data['close'] - 3 * EWMA(np.array(TR), period))), 100) / data['close'],
            np.array((data['close'] + 3 * EWMA(np.array(TR), period))) / np.array((data['close'] - 3 * EWMA(np.array(TR), period)))])

def load_data(asset, path, side = 'train', return_price = False):
    train = pd.read_csv(path + side + '/' + asset + '.csv')
    X, Y = np.array(train)[:, :-2], np.array(train)[:, -2]
    if return_price:
        return X, Y, np.array(train)[:,-1]
    return X, Y

def to_onehot(inp):
    B, S, H = np.where(inp == -1)[0], np.where(inp == 1)[0], np.where(inp == 0)[0]
    onehot = np.zeros((inp.shape[0], 3))
    onehot[B], onehot[H], onehot[S] = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    return onehot
    
def eval_label(data, chunkSize):
    chunks = [data[x : x + chunkSize] for x in range(0, len(data), chunkSize)]
    high, low = np.array([np.max(chunk) for chunk in chunks]), np.array([np.min(chunk) for chunk in chunks])
    tot_gain, mean_gain = np.sum(high/low), np.mean(high/low)
    tot_profit, mean_profit = np.sum(high - low), np.mean(high - low)
    return tot_gain, mean_gain, tot_profit, mean_profit
