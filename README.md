# Automated Trading With Reinforcement Learning

*This repository contains code and resources for automated trading using reinforcement learning methods. The content provided here is for educational and informational purposes only. It is not intended to be a financial product or investment advice.*

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

#### Motivation and Approach

In the domain of stock price prediction and trading algorithm implementation, there are numerous examples available on platforms such as GitHub. However, the majority of these implementations fail to deliver consistent performance. Many machine learning-based approaches rely on non-linear regression methods, such as Long Short-Term Memory (LSTM) networks or transformers, to directly forecast future prices. This approach is inherently flawed, as stock prices are highly dynamic and cannot be effectively modeled as a simple function of time.

Moreover, direct price forecasting does not offer actionable insights for determining optimal entry and exit points in trading. While more sophisticated methods attempt to predict price trends, these still provide limited utility for making precise trade decisions.

To create real value for trading, a predictive model must generate clear signals indicating favorable conditions for entering or exiting trades. Additionally, given the constantly evolving nature of financial markets, any predictive model must be capable of adapting to changing market conditions in real-time. This project addresses these challenges using reinforcement learning, specifically policy gradient methods.

#### Intuition

Reinforcement learning is particularly well-suited for the task of developing a robust trading strategy. For those unfamiliar, the fundamentals of reinforcement learning are as follows:

* **Optimal Action Discovery** 

    Unlike traditional supervised learning methods, where the optimal action for each state is known, reinforcement learning excels in scenarios where the optimal action is unknown. Instead, the effect of taking an action in a given state can be estimated by a value or reward.

* **Training Dynamics** 

    Reinforcement learning typically involves training in episodes rather than epochs, as actions taken by the agent influence subsequent states. The primary objective is to maximize the cumulative reward achieved during each training episode.

* **Application to Financial Markets** 

    In the context of stock or asset prices, the state represents the current market conditions (and potentially the condition of the trading portfolio). For each state, the actions could be decisions such as buy, sell, or hold (or their variations). The reward is defined by the total gain (or loss) resulting from the actions taken.

This approach allows the model to continuously learn from market interactions, improving its decision-making process over time to maximize returns. This is in contrast to traditional models that may struggle with the dynamic and uncertain nature of financial markets.

For more details, see [here](reinforce/README.MD)

## Live Trading
...

## Usage
...

## Testing
...

## Contribution
...