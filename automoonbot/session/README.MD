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
            }
        })

    def close_position(self, name, price):
        position = positions.pop(name)
        position["exit"] = price

    def position_value(self, name, price):
        position = positions.get(name)
        return position["size"] * (price / position["entry"])
```

This implementation would work fine for most cases. It creates an easy way to open, close positions, and to compute the current value of any open positios. However, if a large number of operations need to be performed simultaneously, the execution time grows linearly. Of course, modules suck as `multiprocessing` can be used to parallelize the operations, but it also adds a lot of complexity to the design and possibility for errors.

One great property for the positions to have is additivity, such that opening a new position is simply represented as `portfolio += position`, and closing a position `portfolio -= position`. To achieve that, we can represent positions as tensors.

## Tensor Representation

Let's represent a single position as a matrix, where the row indicates the asset type (*e.g. equity, currency ...*), and columns representing the specific properties associated with the position (*e.g. size, value ...*). Then the entire portfolio can be expressed as an array of positions, or a 3D tensor with axes `[num_positions, position_types, position_attrs]`.

The cash balance of the portfolio can be represented as a `currency` position, specifically in `USD` for most viewers. To represent a trade, for instance 'buying $100 worth of TSLA stock', we first decompose it into two operations:

1. Take $100 off cash balance:

    ```
    t1 = empty_portfolio_tensor
    t1[balance_row, USD, [size]] = [-100]
    ```

2. Convert the $100 to the equivalent amount of TSLA stock

    ```
    t2 = empty_portfolio_tensor
    t2[some_row, TSLA, [size, value]] = [TSLA_price / 100, TSLA_price]
    ```

Then, the entire trade can simply be represented as

```
t = t1 + t2
```

Finally, to execute the trade

```
portfolio += t
```