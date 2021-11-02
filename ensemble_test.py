import numpy as np
import matplotlib.pyplot as plt
from subroutines import *
from ensemble import ensembled

def enter(prediction, price, risk):
    takeProfit = 0
    stopLoss = float('inf')
    out = np.zeros(len(price))
    i = 0
    state = 0
    while i < len(price):
        if state == 0:
            if prediction[i] == -1:
                state = 1
                takeProfit = price[i] * (1 - 3 * risk)
                stopLoss = price[i] * (1 + risk)
        elif state == 1:
            if price[i] > stopLoss or price[i] < takeProfit:
                out[i] = -1
                state = 0
            elif prediction[i] == -1:
                takeProfit = price[i] * (1 - 3 * risk)
                stopLoss = price[i] * (1 + risk)
        i += 1
    return out

def exit(prediction, price, risk):
    takeProfit = float('inf')
    stopLoss = 0
    out = np.zeros(len(price))
    i = 0
    state = 0
    while i < len(price):
        if state == 0:
            if prediction[i] == 1:
                state = 1
                takeProfit = price[i] * (1 + 3 * risk)
                stopLoss = price[i] * (1 - risk)
        elif state == 1:
            if price[i] < stopLoss or price[i] > takeProfit:
                out[i] = 1
                state = 0
            elif prediction[i] == 1:
                takeProfit = price[i] * (1 + 3 * risk)
                stopLoss = price[i] * (1 - risk)
        i += 1
    return out

def visualize(price, prediction):
    B, S = np.where(prediction == -1)[0], np.where(prediction == 1)[0]
    plt.figure(figsize = (16, 8))
    plt.title('Prediction')
    plt.plot(price, 'c', linewidth = 0.5)
    plt.plot(B, price[B], '^', c = 'red')
    plt.plot(S, price[S], 'v', c = 'green')
    plt.legend(['Closing Price', 'Buy', 'Sell'])
    plt.show()
    
def step(price, predictions):
    B = enter(predictions, price, 0.02)
    S = exit(predictions, price, 0.02)
    B, S = np.where(B == -1)[0], np.where(S == 1)[0]
    pred = np.zeros(len(predictions))
    pred[B], pred[S] = -1, 1
    visualize(price, pred)

if __name__ == "__main__":
    X, Y, price = load_data('BTC', 'data/', return_price = True)
    model = ensembled('BTC')
    pred = model.predict(X)
    step(price, pred)