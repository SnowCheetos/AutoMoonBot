from legacy.trader import Trader
from legacy.trainer import Trainer
import matplotlib.pyplot as plt
from legacy.utils import generate_plot
from collections import deque

plt.ion()  # Turn on interactive mode

fig, ax = plt.subplots(2, 1, figsize=(12, 6))
#plt.tight_layout()
plt.subplots_adjust(hspace = 0.5)

# Creating empty lines
line1, = ax[0].plot([], [])
line2, = ax[1].plot([], [])

ax[0].set_ylabel("Net Worth ($)")
ax[0].set_xlabel("Time (hours)")
ax[0].set_title("Simulated Net Worth Over Time")

ax[1].set_ylabel("Market Price ($)")
ax[1].set_xlabel("Time (hours)")
ax[1].set_title("Market & Action Tracker")

#max_len = 300
networth_hist = [] #deque(maxlen=max_len)
price_hist = [] #deque(maxlen=max_len)
take_profit_line = None
stop_loss_line = None

if __name__  == "__main__":
    trainer = Trainer()
    trader = Trader("BTC/USDT", trainer)
    trader.DELAY = 0
    
    max_runs = 10_000
    counter = 0
    while counter < max_runs:
        train = False
        if counter % 50 == 0:
            train = True
        n, p, a = trader.step(train=train)
        counter += 1

        networth_hist.append(n)
        price_hist.append(p)

        take_profit_line, stop_loss_line = generate_plot(counter, fig, ax, line1, line2, networth_hist, price_hist, trader.take_profit, trader.stop_loss, take_profit_line, stop_loss_line, [counter, a])