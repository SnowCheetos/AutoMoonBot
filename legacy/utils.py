import numpy as np
import matplotlib.pyplot as plt

def label_data(data, chunk_size):
    labels = np.ones(data.shape[0])
    chunks = data[:data.shape[0]//chunk_size*chunk_size, 3].reshape(-1, chunk_size)
    mins, maxs = chunks.argmin(1), chunks.argmax(1)
    buy_idx = np.arange(len(mins)) * chunk_size + mins
    sell_idx = np.arange(len(maxs)) * chunk_size + maxs
    labels[buy_idx] = 0
    labels[sell_idx] = 2

    remaining = data[data.shape[0]//chunk_size*chunk_size:, 3]
    if remaining.size > 0:
        min_idx = np.argmin(remaining)
        max_idx = np.argmax(remaining)
        labels[data.shape[0]//chunk_size*chunk_size + min_idx] = 0
        labels[data.shape[0]//chunk_size*chunk_size + max_idx] = 2

    return labels


def calculate_rsi(data, period):
    closing_prices = data[:, 3]
    price_diff = np.diff(closing_prices)
    
    gains = np.where(price_diff > 0, price_diff, 0)
    losses = np.where(price_diff < 0, -price_diff, 0)
    
    avg_gain = np.empty(len(price_diff))
    avg_loss = np.empty(len(price_diff))
    
    avg_gain[:period] = np.nan
    avg_loss[:period] = np.nan
    
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    for i in range(period+1, len(price_diff)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

    rs = avg_gain / avg_loss
    rsi = 0.5 - (1 / (1 + rs))

    rsi = np.insert(rsi, 0, np.nan)

    return rsi

def calculate_sma(data, period):
    closing_prices = data[:,3]

    sma = np.convolve(closing_prices, np.ones(period), 'valid') / period
    sma = np.concatenate((np.full(period-1, np.nan), sma)) / closing_prices

    return sma - 1

def calculate_ema(data, period):
    closing_prices = data[:,3]

    ema = np.empty_like(closing_prices)
    ema[:period] = np.nan
    ema[period] = np.mean(closing_prices[:period])

    multiplier = 2 / (period + 1)
    for i in range(period+1, len(closing_prices)):
        ema[i] = (closing_prices[i] - ema[i-1]) * multiplier + ema[i-1]

    ema = ema / closing_prices

    return ema - 1

def calculate_bollinger_bands(data, period, k=2):
    closing_prices = data[:,3]
    
    sma = np.convolve(closing_prices, np.ones(period), 'valid') / period
    sma = np.concatenate((np.full(period-1, np.nan), sma))

    std_dev = np.array([np.std(closing_prices[i-period+1:i+1]) for i in range(period-1, len(closing_prices))])
    std_dev = np.concatenate((np.full(period-1, np.nan), std_dev))

    #upper_band = sma + k * std_dev
    #lower_band = sma - k * std_dev

    bb_value = (closing_prices - sma) / (k * std_dev + 1e-10)
    
    return bb_value #, upper_band, lower_band

def calculate_stochastic_oscillator(data, period):
    high_prices = data[:,1]
    low_prices = data[:,2]
    closing_prices = data[:,3]

    highest_high = np.array([np.max(high_prices[i-period+1:i+1]) for i in range(period-1, len(high_prices))])
    highest_high = np.concatenate((np.full(period-1, np.nan), highest_high))

    lowest_low = np.array([np.min(low_prices[i-period+1:i+1]) for i in range(period-1, len(low_prices))])
    lowest_low = np.concatenate((np.full(period-1, np.nan), lowest_low))

    k_value = 100 * ((closing_prices - lowest_low) / (highest_high - lowest_low))

    k_value_normalized = (k_value - 50) / 50

    return k_value_normalized


def generate_plot(i, fig, ax, line1, line2, nw_hist, p_hist, tp, sl, take_profit_line, stop_loss_line, action=None):
    line1.set_ydata(nw_hist)
    line1.set_xdata(range(len(nw_hist)))
    
    line2.set_ydata(p_hist)
    line2.set_xdata(range(len(p_hist)))
    
    ax[0].relim()
    ax[0].autoscale_view()

    ax[1].relim()
    ax[1].autoscale_view()

    if tp:
        if take_profit_line:  # If the line already exists
            take_profit_line.remove()  # Remove it
        take_profit_line = ax[1].axhline(y=tp, color='g', linestyle='--')  # Draw a new line
        take_profit_line.set_label('Take-profit')  # Set the label for the legend
    if sl:
        if stop_loss_line:
            stop_loss_line.remove()
        stop_loss_line = ax[1].axhline(y=sl, color='r', linestyle='--')
        stop_loss_line.set_label('Stop-loss')

    if action[1] is not None:
        idx, d = action
        if d == -1:
            ax[1].scatter(idx, p_hist[-1], marker="^", c="r")
        elif d == 1:
            ax[1].scatter(idx, p_hist[-1], marker="v", c="g")

    #ax[1].legend()
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)
    if i % 20 == 0:
        plt.savefig(f"media/frames/{i}.png")

    return take_profit_line, stop_loss_line
