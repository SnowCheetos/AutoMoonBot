import numpy as np
from enum import Enum
from typing import List, Dict


class Portfolio:
    """
    An object representing the portfolio of one session.
    Portfolio value is unitless between 0 to infinity, representing ratio.
    High precision float ops are required.
    All floats use `np.float64` for numeric stability.
    All float ops use `numpy` for the same reason.
    """

    class ColAttr(Enum):
        Value = 0
        LogQuote = 1
        LagQuote = 2

    def __init__(
        self,
        fiat: str,
        tradables: List[str],
    ) -> None:
        self.index_map = {t: i for t, i in enumerate(tradables)}
        self.fiat = self.index_map[fiat]
        self._portfolio = self._reset_portfolio(self.fiat, len(tradables))

    def _reset_portfolio(
        self,
        fiat: int,
        rows: int,
    ) -> np.ndarray:
        portfolio = np.zeros(
            (
                rows,
                len(
                    self.__class__.ColAttr,
                ),
            ),
            dtype=np.float64,
        )
        portfolio[
            :,
            [
                self.__class__.ColAttr.LogQuote,
                self.__class__.ColAttr.LagQuote,
            ],
        ] = [1.0, 1.0]
        portfolio[fiat, self.__class__.ColAttr.Value] = 1.0
        return portfolio

    def _reset_lag(self) -> None:
        self._portfolio[
            :,
            self.__class__.ColAttr.LagQuote,
        ] = self._portfolio[:, self.__class__.ColAttr.LogQuote]

    @property
    def U(
        self,
        diag: bool = False,
    ) -> np.ndarray:
        u = np.exp(
            np.diff(
                self._portfolio[
                    :,
                    [
                        self.__class__.ColAttr.LagQuote,
                        self.__class__.ColAttr.LogQuote,
                    ],
                ]
            )
        )[:, 0]
        self._reset_lag()
        if diag:
            return np.diag(u)
        return u

    def update_quotes(
        self,
        quotes: Dict[str, float],
    ) -> None:
        keys = list(quotes.keys())
        index = [self.index_map[key] for key in keys]
        value = [np.log(quotes[key]) for key in keys]
        self._reset_lag()
        self._portfolio[
            index,
            self.__class__.ColAttr.LogQuote,
        ] = value

    def _value_transfer(
        self,
        src: List[int] | int,
        tgt: List[int] | int,
        size: np.ndarray | np.float64,
    ) -> np.ndarray:
        t = self.U(diag=True)
        t[src, src] -= size
        t[src, tgt] += size
        return t

    def _build_transaction(
        self,
        transactions: Dict[str, float],
    ) -> np.ndarray:
        src, tgt, size = [], [], []
        for transaction in transactions:
            if transaction["type"] == "buy":
                src.append(self.fiat)
                tgt.append(self.index_map[transaction["asset"]])
                size.append(transaction["size"])
            elif transaction["type"] == "sell":
                src.append(self.index_map[transaction["asset"]])
                tgt.append(self.fiat)
                size.append(transaction["size"])
        size = np.array(size, dtype=np.float64)
        return self._value_transfer(src, tgt, size)

    def apply_transaction(
        self,
        transaction: np.ndarray,
    ) -> None:
        self._portfolio[
            :,
            self.__class__.ColAttr.Value,
        ] = (
            self._portfolio[
                :,
                self.__class__.ColAttr.Value,
            ]
            @ transaction
        )
