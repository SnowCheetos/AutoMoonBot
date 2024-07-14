import numpy as np

# TODO I need to spend some time thinking about how to do this...
class Portfolio:
    """
    An object representing the portfolio of one session.
    Portfolio value is unitless between 0 to infinity, representing ratio.
    High precision float ops are required.
    All floats use `np.float64` for numeric stability.
    All float ops use `numpy` for the same reason.
    May move to Fortran to utilize higher precision floats than `np.float64`.

    Positions (trades) represented as numpy array.
    Rows are used to index a specific position.
    Columns are used to index specific fields.
    """

    # Row Indices
    _balance_index: int = 0

    # Col Indices
    _cols: int = 8  # Total number of columns
    _time_index: int = 0  # Float timestamp for creation time
    _type_index: int = 1  # Type of position, e.g. currency
    _subtype_index: int = 2  # Subtype of position, e.g. USD
    _size_index: int = 3  # Position size, in ratio of portfolio
    _entry_index: int = 4  # Asset price at entry
    _margin_index: int = 5  # The amount of margin used, default 1
    _exchange_index: int = 6  # Whether or not it can be traded currently
    _expire_index: int = 7  # When the position expires, 0 for None

    # Position Types
    _currency: np.float64 = 0
    _equity: np.float64 = 1
    _crypto: np.float64 = 2
    _options: np.float64 = 3
    _commodities: np.float64 = 4

    def __init__(
        self,
        timestamp: np.float64,
        init_balance: np.float64 = 1.0,
        max_positions: int = 100,
    ) -> None:
        self._initialize_positions(timestamp, init_balance, max_positions)

    def reset(
        self,
        timestamp: np.float64,
        init_balance: np.float64 = 1.0,
        max_positions: int = 100,
    ) -> None:
        self._initialize_positions(timestamp, init_balance, max_positions)

    def _initialize_positions(
        self, timestamp: np.float64, init_balance: np.float64, max_positions: int
    ) -> None:
        self._positions = np.zeros(
            (max_positions, self.__class__._cols),
            dtype=np.float64,
        )
        self._open_slots = set(range(max_positions))
        self._open_slots.pop(self.__class__._balance_index)
        self._positions[
            self.__class__._balance_index,
            [
                self.__class__._time_index,
                self.__class__._type_index,
                self.__class__._subtype_index,
                self.__class__._size_index,
                self.__class__._entry_index,
                self.__class__._margin_index,
                self.__class__._exchange_index,
                self.__class__._expire_index,
            ],
        ] = np.array(
            [
                timestamp,
                self.__class__._currency,
                np.float64(hash("usd")),
                init_balance,
                1.0,
                1.0,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )

    def open_position(
        self,
        timestamp: np.float64,
        ptype: np.float64,
        subtype: np.float64,
        size: np.float64,
        entry: np.float64,
        margin: np.float64 = 1.0,
        exchange: np.float64 = 1.0,
        expire: np.float64 = 0.0,
    ) -> int:
        if not self._open_slots:
            return -1
        slot = self._open_slots.pop()
        self._positions[
            slot,
            [
                self.__class__._time_index,
                self.__class__._type_index,
                self.__class__._subtype_index,
                self.__class__._size_index,
                self.__class__._entry_index,
                self.__class__._exchange_index,
                self.__class__._margin_index,
                self.__class__._expire_index,
            ],
        ] = np.array(
            [
                timestamp,
                ptype,
                subtype,
                size,
                entry,
                margin,
                exchange,
                expire,
            ],
            dtype=np.float64,
        )
        if ...: # TODO Checks if we have enough money first
            pass
        return slot

    def close_position(self):
        pass
