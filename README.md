# Automated Trading With Reinforcement Learning

This repository contains code and resources for automated trading using reinforcement learning methods. The content provided here is for educational and informational purposes only. It is not intended to be a financial product or investment advice.

## Example Demonstration
![](media/dem.gif)

Shown above is a demonstration of system back test. Data used for this test was *SPDR S&P 500 ETF Trust (SPY)* with *1 hour* interval, starting *2022/06/17*. This was an interesting period as it represented a recent bear market.

## Disclaimer

1. **Not Financial Advice:** The information and tools provided in this repository should not be interpreted as financial advice or recommendations. The strategies, models, and code are provided for educational purposes only.
2. **No Warranty:** The content and information are provided "as is," without any warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement.
3. **Risk of Loss:** Trading financial instruments involves risk, and you could lose money. You should carefully consider your investment objectives, level of experience, and risk appetite before using the tools provided in this repository.
4. **No Liability:** The creators and contributors of this repository are not liable for any financial losses or damages that may occur through the use of the information or tools provided. You are solely responsible for any financial decisions you make based on this information.
5. **Professional Advice:** Always seek the advice of a qualified financial professional before making any investment decisions.

By using the code and resources provided in this repository, you acknowledge and agree to this disclaimer.

## Introduction

There are countless examples of stock price predictors and trade bot implementations on GitHub, rarely do any of them work consistently. A majority of the machine learning based implementations attempt to use non-linear regression methods such as *LSTM* or *transformers* to forecast future prices directly, which is unlikely to succeed since price ranges are highly dynamic and cannnot be easily expressed as a function of time. 

Furthermore, directly forecasting prices does not provide information when assessing entry and exit points for trade placements. More sophisticated methods aim to forecast the price trend, which while more valuable than direct price forecast, still only provides limited value for trade placements.

To provide value for trade placements, the predictor should be able to produce signals when the conditions are favorable for entry or exit. On top of that, due to the dynamic nature of the market, a predictor of any kind **must** be able to adapt to the changing market condition effectively. This project aims to solve these issues with *reinforcement learning* (or more specifically, *policy gradient*). 

## Intuition

Reinforcement learning is a natural choice for this task, for those unfamiliar, the basics of it are as follows:

* Unlike traditional supervised learning methods, in which the optimal target action for each state is *known*, reinforcement learning excels when the optimal target action at each state is *unknown*, but the affect of taking an action at a given state can be estimated by a *value* or *reward*.

* Reinforcement learning typically train in episodes as opposed to epochs, since actions can affect the state. The goal is to maximize reward achieved for each training episode.

* In the case for stock or asset prices, the state would be the current condition of the market (additionally the condition of the portfolio), and the for each state would be whether to *buy, sell or hold* (or variations of it), and the reward would be the total gain (or loss).

For more details, see [here](reinforce/README.MD)

## Prediction Processing
...

## Live Trading
...

## Installation
...

## Usage
...

## Testing
...