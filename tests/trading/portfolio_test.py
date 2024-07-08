import pytest
from trading import Portfolio, Category, Status


def test_portfolio():
    portfolio = Portfolio(100, 0)

    opened = portfolio.open_positions()
    value = portfolio.value()

    assert len(opened) == 0, 'no open positions yet'
    assert value == 1, 'portfolio incorrectly computed value'

    status = portfolio.open(Category.LONG, 0.5)
    opened = portfolio.open_positions()
    value = portfolio.value()

    assert status == Status.SUCCESS, 'portfolio did not open position correctly'
    assert len(opened) == 1, 'portfolio did not properly add position'
    assert value == 1, 'portfolio incorrectly computed value'

    portfolio.update(200)
    opened = portfolio.open_positions()
    value = portfolio.value()

    assert status == Status.SUCCESS, 'portfolio did not open position correctly'
    assert len(opened) == 1, 'portfolio did not properly add position'
    assert value == 1.5, 'portfolio incorrectly computed value'