# Automated Trading with Policy Gradient

Please review the disclaimers [`here`](DISCLAIMER.md) before proceeding. By continue visiting the rest of the content, you agree to all the terms and conditions of the disclaimers.

## Introduction

*Please note that this is a documentation for legacy code, which as of July 2024 is no longer being actively maintained or supported. Features are not guaranteed to function properly, use for insights and educational purposes only. For more up-to-date developments, check out the current version.*

### Demonstration

![](media/demo.gif)

- The gif might take some time to load.

- Demonstration of system back-testing. Data used for this test was *SPDR S&P 500 ETF Trust (SPY)* with *1 hour* interval, with a start date in *June 2022.*

### Motivation and Approach

In the domain of stock price prediction and trading algorithm implementation, there are numerous examples available on platforms such as GitHub. However, the majority of these implementations fail to deliver consistent performance. 

Many machine learning based approaches rely on non-linear regression methods, particularily *autoregressive* models such as Long Short-Term Memory (*LSTM*) networks and *Transformers*, to directly forecast future prices. This approach is inherently flawed, as stock prices are highly dynamic and cannot be effectively modeled as a simple function of time.

Moreover, direct price forecasting does not offer actionable insights for determining optimal entry and exit points in trading. While more sophisticated methods attempt to predict price trends using statistical methods such as *autocorrelation* and *STL-Decomposition*, these still provide limited utility for making precise trade decisions.

To create real value for trading, a predictive model must generate clear signals indicating favorable conditions for entering or exiting trades. Additionally, given the constantly evolving nature of financial markets, any predictive model must be capable of adapting to changing market conditions in *real-time*. This project addresses these challenges using reinforcement learning, specifically *policy gradient* methods.

### Intuition

Reinforcement learning is particularly well-suited for the task of developing a robust trading strategy. For those unfamiliar, `tensorflow` provides a comprehensive [`tutorial`](https://www.tensorflow.org/agents/tutorials/0_intro_rl) with examples. Here are some reasons why reinforcement learning suits financial markets particularily well.

* **Optimal Action Discovery** 

    Unlike traditional supervised learning methods, where the optimal action for each state is known, reinforcement learning excels in scenarios where the optimal action is *unknown*. Instead, the effect of taking an action in a given state can be estimated by a value or reward.

    - *e.g.* It's pretty difficult to train a model to be profitable at trading, but it's quite easy to tell if a model is mooning or aping simply by sitting aside and judge every trade it makes.

* **Training Dynamics** 

    Reinforcement learning typically involves training in episodes rather than epochs, as actions taken by the agent influence subsequent states. The primary objective is to maximize the *cumulative reward* achieved during each training episode.

* **Application to Financial Markets** 

    In the context of stock or asset prices, the state represents the current market conditions (*and potentially the condition of the trading portfolio*), the actions could be decisions such as buy, sell, or hold (*or something similar*), which would affect any future states. The reward could be defined by the total gain (*or loss*) resulting from the actions taken.

This approach allows the model to continuously learn from market interactions, improving its decision-making process over time to maximize returns. This is in contrast to traditional models that may struggle with the dynamic and uncertain nature of financial markets. For more details, see [`reinforce/README.MD`](reinforce/README.MD)

### Configuration

Most configurations can be done by modifying fields in [`config.json`](config.json), a few fields are worth pointing out:

- `backtest_interval` is the rate of which data is read from the database and displayed on the UI, `1s` means it'll fetch one row of data per physical second and perform inference, regardless of the actual price data interval present in the dataset.

- `queue_size` is the maximum number of rows to keep in memory at any given point, it is required for technical indicator and statistical descriptor computations. 

    - Make sure the value is more than the maximum value present in `feature_params`.

- `action_cost` and `inaction_cost` represent the reward or punishment of taking any action, they have a significant impact on the model performance. 

    - Generally, a larger `action_cost` will result in more conservative sequence of actions (*e.g. more hold, less buy/sell*), opposite that of `inaction_cost`.

- `sharpe_cutoff` [sharpe ratio](https://www.investopedia.com/terms/s/sharperatio.asp) is used in reward computation, this field determines how many previous positions within the episode to use when computing the sharpe ratio.

- `alpha` is the take-profit to stop-loss ratio, adjust base on your risk tolerance.

- `beta`, `gamma` and `zeta` are parameters used when computing the loss as follows, these parameters have been observed to have very significant impact on the model performance

    $$L(\theta)=-\sum_{t=0}^{T}\log\pi_{\theta}(a_t|s_t) \cdot R_t$$

    $$R_t = \beta \cdot \bar{r}_t + (1-\beta) \cdot G_T$$

    $$G_T = \log {P_{t=T} \over P_{t=0}}$$

    - $t$ is the time step, interpreted as each [`OHLCV`](https://en.wikipedia.org/wiki/Open-high-low-close_chart) candle, in unit of price data interval

    - $T$ is the step at which the episode ended

    - $\log \pi_{\theta}(a_t|s_t)$ is the log probability of action $a_t$ at state $s_t$ under policy $\pi$ parameterized by $\theta$

    - $P_t$ represents the portfolio value at time $t$
    
    - $G_T$ represents the total portfolio log return for the entire episode.
    
        - *Although intuitively it makes more sense to add it after the summation, but it was determined empirically that this expression produced better results*

    - $\beta$ is the parameter of interest, it represents the importance of log return when computing loss, which is ignored when set to `null`.

    - $\bar{r}_t$ is the normalized discounted reward received at step $t$, which when $a_t = 2$ (*e.g. sell*) is computed as

        - Note that in this case, using the log return might make more sense than $R-1$

    $$r_t = (1 - \zeta) S_t + \zeta \left({P_{t} \over P_{\arg\max_{t} ( t \mid a_t = 0 )}} - 1\right) W_d$$

    - $W_d$ represents the reward multiplier for doubling, more on that later when discussing `full_port`
    
    - $\arg\max_{t} ( t \mid a_t = 0 )$ represents the last time $t$ where $a_t=0$ (*e.g. buy*), or when the trade was first opened
    
    - $S_t$ represents the sharpe ratio of the all the trades up until time step $t$, whose importance is described by parameter `zeta` $\zeta$. The sharpe ratio $S_t$ is computed as follows

    $$S_t = { {R_{P(t)} - R_f} \over \sigma_{P(t)} }$$

    - $R_{P(t)}$ represents the cumulative portfolio return from $t=0$ up to time step $t$
    
    - $\sigma_{P(t)}$ represents the standard deviation of the portfolio up until time step $t$

    - $R_f$ represents the concept of [*risk free return*](https://www.investopedia.com/terms/r/risk-freerate.asp) for the same time period, in this case it uses the *buy-and-hold* return value on the same asset

    $$g_t = \sum_{t=k}^{T} \gamma^{t-k} r_t$$

    $$\bar{r}_t = { { g_t - \mu_r } \over \sigma_r }$$

    - $\gamma$ is the discount rate, which exponentially drops off as it approaches infinite future

- For more details on the reward and loss computations, see [`reinforce/environment.py`](reinforce/environment.py#L156) and [`reinforce/model.py`](reinforce/model.py#L112)

- `record_frames` when set to `true` will save every frame of the backtest.

- `feature_params` feature computation parameters. For more details, see [`descriptors.py`](utils/descriptors.py)

- `retrain_freq` determines how often the model retrains itself to adjust for new environments, in unit of `backtest_interval`.

- `inference_method` one of `[prob|argmax]`. When set to `prob`, actions will be sampled from the model output distribution, when set to `argmax`, the action with the highest probability will always be chosen. In general, `prob` produces better results but are less consistent and reproducible.

- `full_port` describes whether or the model should go full ape mode, if `true`, it will use $100 \%$ of the portfolio for each trade, otherwise the amount of portfolio to use is determined by the action probability.

    - When set to `false`, the model will automatically enable [`doubling`](backend/manager.py#L67) (*similar to Blackjack*), which allows the model to *double down* on its existing position once more if an opportunity appears. 

        - *e.g.* If the model used $50\%$ of the portfolio value to bought a stock at $\$100$ per share, and the price decreased to $\$90$ per share, the model could decide to double down and use the rest $50\%$ to buy at $\$90$ per share, effectively lowering the overall entry cost to $\$95$ per share by *doubling* its position size.

        - If the price later rises to $\$98$ per share and the model decides to sell, without doubling, the trade itself would have made a loss of $-2\%$, and the portfolio would have lost $-1\%$ of its value (*since the trade used $50\%$ of the portfolio value*), but with doubling, both the trade and the portfolio would have gained $+3.16\%$.

        - Consequently, if the price continued to drop, say to $\$80$ per share, then without doubling, the trade would have made a loss of $-20\%$, and for the portfolio, $-10\%$. However, if doubled, both the trade and portfolio would have made a loss of $-15.79\%$. 

    - To account for the increased potential risk and reward, the reward multiplier $W_d$ is set to $2$ when doubling is in effect, otherwise $1$, so the model would receive $2\times $ the reward or punishment for the action.

    - Experiments showed that when `full_port` is enabled, it generally out-performs the doubling model, and often times manages to beat the market, even in relatively long timeframes. 

    - On the other hand, when `full_port` is disabled, the model generally produces more stable portfolios, and very consistently stays positive during bear markets.

    - For more details, see [`utils/trading.py`](utils/trading.py#L114)


## Usage

*Tested with Python 3.10 on Ubuntu 22.04 and MacOS 14.5, not guaranteed to work on Windows machines.*

### Basics
First, install python and requirements, using [`anaconda`](https://www.anaconda.com/) is recommended.

```sh
pip install -r requirements.txt
```

Then, launch the server

```sh
python app.py
```

If everything went well, you should be able to access `http://localhost:29697` in a browser and visualize the back testing take place.

As of now, only back testing is supported. Two datasets are made available with the repo (*SPY and NVDA 1hr 2y*). To change the dataset, open `config.json` and change the `ticker` field to the ticker of your choice. 

To download new datasets, go to `utils/download_data.py` and modify the script, and run it from the project root. Then, go to `data` directory and modify `helper.json` as shown.

## Testing

Test scripts are available in the `tests` directory, once `pytest` is installed, run a test by going into the `tests` directory and execute `pytest <test_script>.py`. Note that some tests might be outdated.