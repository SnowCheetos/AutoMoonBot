# Portfolio

To simulate trading, a well designed portfolio simulator is needed.

## Intuition

A simple way to implement a portfolio would be to use an class

```python
class Portfolio:
    positions = dict()

    def open_position(self, name, size, price):
        positions.update({
            name: {
                "size": size,
                "entry": price,
                "price": price,
                "exit": None,
            }
        })

    def update_position(self, name, price):
        positions[name].update({
            "price": price,
        })

    def close_position(self, name, price):
        position = positions.pop(name)
        position["exit"] = price

    def position_value(self, name):
        position = positions.get(name)
        return position["size"] * (
            position["price"] / position["entry"]
        )
```

This implementation would work fine for most cases. It creates an easy and intuitive way to open, close, and update positions. However, if a large number of operations need to be performed in quick succession or simultaneously (*which is the intention*), the performance can quickly become untenable. Of course, modules such as `multiprocessing` can be used to parallelize the operations, but it also adds a lot of complexity to the design and possibility for errors. This approach also comes with the added downside of needing to track and update a number of attributes for each position, which can get extremely tedious.

## Matrix Representation

Like many other things in life, a portfolio can be represented by matrices and would allow us to use matrix operations to execute transactions, which could bring significant benefits to the performance, especially when it comes to a large number of frequent transactions.

For this purpose, we represent a portfolio as matrix $`\mathbf{P} \in \mathbb{R}^{n \times m}`$, where

- $n$ is the number of rows, each row represents an unique tradable asset, which also includes the fiat currency

- $m$ is the number of columns, each column represents an attribute for the asset

As an example, say we want to follow assets $\textbf{USD}$, $\textbf{BTC}$, $\textbf{AMD}$, $\textbf{SPY}$, where $\textbf{USD}$ is the fiat currency. The row index for each asset follows the same order. We then define the following attributes for each asset

1. $\textbf{value}$
    - Current value of the asset within the portfolio, in unit of fiat currency.

2. $\textbf{logQuote} = \log(\text{p}_t)$
    - The most recent market price quote for the asset, in $\log$ scale.

3. $\textbf{lagQuote} = \log(\text{p}_{t-1})$
    - The previous market price quote for the asset, also in $\log$ scale.

If we start with $`\$1000`$ and hold no other assets, then we can initialize the matrix $\mathbf{P}$ as follows

$$
\mathbf{P}= 
\begin{array}{c|ccc}
  & \text{value} & \log(\text{p}_t) & \log(\text{p}_{t-1}) \\
\hline
\text{USD} & 1000 & 0 & 0 \\
\text{BTC} & 0 & 0 & 0 \\
\text{AMD} & 0 & 0 & 0 \\
\text{SPY} & 0 & 0 & 0 \\
\end{array}
$$

The first thing we need to do is update the prices with some data, note since $\textbf{USD}$ is in unit of the fiat currency itself, its price value will always be $1$, and since $`\log(1)=0`$, the $\textbf{logQuote}$ and $\textbf{lagQuote}$ attributes for $\textbf{USD}$ will stay constant at $`0`$

Say $\textbf{BTC}$ last traded at $`\$50,000`$, $\textbf{AMD}$ at $`\$200`$, and $\textbf{SPY}$ at $`\$500`$, then

$$
\mathbf{P} = 
\begin{bmatrix}
1000 & 0 & 0 \\
0 & \log(50000) & 0 \\
0 & \log(200) & 0 \\
0 & \log(500) & 0 \\
\end{bmatrix}
$$

If we assume that we can fetch new data at each time step $t$, then we can write the data update loop as

$$
\begin{aligned}

& \textbf{for} \, t \in \tau \, \textbf{do} \\

& \quad \textbf{for} \, \text{asset} \in \text{assets} \, \textbf{do} \\

& \qquad \text{quote}_{\text{asset}} \leftarrow \text{GET}(\text{asset}, t) \\

& \qquad \mathbf{P}_{\text{asset}, \text{lagQuote}} \leftarrow \mathbf{P}_{\text{asset}, \text{logQuote}} \\

& \qquad \mathbf{P}_{\text{asset}, \text{logQuote}}  \leftarrow \log(\text{quote}_{\text{asset}}) \\

& \quad \textbf{end for} \\

& \textbf{end for}

\end{aligned}
$$

Or in other words, at each time step $t$, we get the prices for each asset, and we first set the $\textbf{lagQuote}$ to be the previous $\textbf{logQuote}$, then we set $\textbf{logQuote}$ to the current $`\log`$ price for each asset. Say after one time step, $\textbf{BTC}$ trades at $`\$45,000`$, $\textbf{AMD}$ at $`\$205`$, and $\textbf{SPY}$ at $`\$510`$, then

$$
\mathbf{P} = 
\begin{bmatrix}
1000 & 0 & 0 \\
0 & \log(45000) & \log(50000) \\
0 & \log(205) & \log(200) \\
0 & \log(510) & \log(500) \\
\end{bmatrix}
$$

After at least 1 time step from initialization, we can compute the update matrix as
$$
\begin{aligned}

&\mathbf{U} = 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & \log(45000) - \log(50000) & 0 & 0 \\
0 & 0 & \log(205) - \log(200) & 0 \\
0 & 0 & 0 &\log(510) - \log(500) \\
\end{bmatrix} \\
\\
&\text{Such that for each asset} \\
\\
&\mathbf{U}_\text{asset} = \mathbf{P}_{\text{asset}, \text{logQuote}} - \mathbf{P}_{\text{asset}, \text{lagQuote}}

\end{aligned}
$$

At first glance, all of this may seem confusing and redundant, but in practice, since these are matrix operations, they can all be performed in a single step for all assets

```python
import numpy as np

quotes = get_market_prices(...)

P[:, lag_quote_index] = P[:, log_quote_index]
P[:, log_quote_index] = np.log(quotes)
U = np.diag(np.diff(P[:, [lag_quote_index, log_quote_index]]))
```

Now comes the best part. Transactions such as buying or selling assets can also be expressed as a matrix $\mathbf{T} \in \mathbb{R}^{n \times n}$, where each row represents the source asset, and each column represents the target asset. Say we want to buy $`\$500`$ (*or* $50\%$ *of the cash balance*) worth of $\textbf{AMD}$, then the transaction matrix can be written as

$$
\mathbf{T} = 
\begin{bmatrix}
1-500/1000 & 0 & 500/1000 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

Once the transaction matrix is constructed, it can be executed by multiplying it to the value column of the portfolio matrix

$$
\mathbf{P_v} = 
\begin{bmatrix}
1000\\
0\\
0\\
0\\
\end{bmatrix}
\times
\begin{bmatrix}
-0.5 & 0 & 0.5 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}=
\begin{bmatrix}
500\\
0\\
500\\
0\\
\end{bmatrix}
$$

Simplying by multiplying the portfolio by the transaction matrix, the portfolio updated the correct values for both $\textbf{USD}$ and $\textbf{AMD}$. Even better than that, we can hold multiple transactions in a single matrix for it to be executed at once. For example, if we then want to buy $`\$100`$ worth of $\textbf{BTC}$ and sell $`\$200`$ worth of $\textbf{AMD}$

$$
\mathbf{T} = 
\begin{bmatrix}
1 - 100/500 & 100/500 & 0 & 0 \\
0 & 1 & 0 & 0 \\
200/500 & 0 & 1 - 200/500 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

Then after executating the transaction matrix

$$
\mathbf{P}_\text{value} = 
\begin{bmatrix}
500 \\
0 \\
500 \\
0 \\
\end{bmatrix}
\times
\begin{bmatrix}
1 - 100/500 & 100/500 & 0 & 0 \\
0 & 1 & 0 & 0 \\
200/500 & 0 & 1 - 200/500 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}=
\begin{bmatrix}
600 \\
100 \\
300 \\
0 \\
\end{bmatrix}
$$

As a added bonus, instead of manually computing the position values using initial entry price and current price, we can use the update matrix $\mathbf{U}$ (*after taking the exponent to convert it from log scale to standard scale*) we computed earlier directly, by performing element-wise-multiplication to the transaction matrix first

$$
\mathbf{TU} = \mathbf{T} \circ \exp(\mathbf{U})
$$

Now if $\mathbf{TU}$ is executed, the transactions will be performed, and the values for all assets will automatically be updated as well. At any given time, if we want to obtain the total value of our portfolio, we simply need to sum the $\textbf{value}$ column from $\mathbf{P}$

$$
\textbf{total value} = \sum_{a \in \text{assets}} \mathbf{P}_{a, \text{value}}
$$

When working with the transaction matrix, it's better to think of it as a '*transfer of value*' rather than a financial transaction. Since each row represents the transaction source, and each column represents the target, it can be interpreted as $X\%$ of the cash value was taken away from `source` and added to `target`. To simplify the unit conversions, it's ideal to work with a dimensionless value instead of dollars, or

$$
\begin{aligned}

&\mathbf{P} = 
\begin{bmatrix}
1 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & 0 \\
\end{bmatrix}
\\
\,
\\
&\mathbf{T} = 
\begin{bmatrix}
1-0.5 & 0 & 0.5 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}

\end{aligned}
$$

Now it can be thought of as $\textbf{0.5}$ units of $\textbf{value}$ was transferred from $\textbf{USD}$ to $\textbf{AMD}$. To convert it back to the fiat unit, we simply multiply the values by the initial cash balance.

This set of tools provide very powerful abstractions to portfolio modeling, allowing very efficient executions of transactions, price and value updates to be done.