import pytest
from trading import Position, Positions, Category, Status, Condition

def test_position():
    position = Position(Category.LONG, 100, 0.5)

    assert position.condition == Condition.CREATED, 'position not yet opened'

    status = position.open()

    assert status == Status.SUCCESS, 'position did not successfully open'
    assert position.value() == 0.5, 'position has wrong value upon open'

    position.update(200)

    assert position.value() == 1.0, 'position has wrong value upon update'

    position.update(100)

    status = position.close()

    assert status == Status.SUCCESS, 'position did not successfully close'
    assert position.condition == Condition.CLOSED, 'position did not close'
    assert position.value() == 0.5, 'position has wrong value upon close'

def test_positions():
    positions = Positions(100)

    opened = positions.fetch_opened()
    closed = positions.fetch_closed()

    assert len(opened) == 0, 'positions returned incorrect opened array'
    assert len(closed) == 0, 'positions returned incorrect closed array'

    status = positions.open(Category.LONG, 0.5)
    opened = positions.fetch_opened()
    closed = positions.fetch_closed()
    value = positions.value()

    assert status == Status.SUCCESS, 'position did not properly open'
    assert value == 0.5, 'positions did not properly compute value'
    assert len(opened) == 1, 'positions returned incorrect opened array'
    assert len(closed) == 0, 'positions returned incorrect closed array'

    pid = list(opened.keys())[0]
    position = positions.fetch_one_opened(pid)
    value = position.value()
    assert value == 0.5, 'position did not properly compute value'

    positions.update(200)
    opened = positions.fetch_opened()
    closed = positions.fetch_closed()
    value = positions.value()

    assert value == 1.0, 'positions did not properly compute value'
    assert len(opened) == 1, 'positions returned incorrect opened array'
    assert len(closed) == 0, 'positions returned incorrect closed array'

    positions.open(Category.LONG, 0.5)
    opened = positions.fetch_opened()
    closed = positions.fetch_closed()
    value = positions.value()

    assert value == 1.5, 'positions did not properly compute value'
    assert len(opened) == 2, 'positions returned incorrect opened array'
    assert len(closed) == 0, 'positions returned incorrect closed array'

    status = positions.close(pid)
    opened = positions.fetch_opened()
    closed = positions.fetch_closed()
    value = positions.value()

    assert status == Status.SUCCESS, 'position did not properly close'
    assert value == 0.5, 'positions did not properly compute value'
    assert len(opened) == 1, 'positions returned incorrect opened array'
    assert len(closed) == 1, 'positions returned incorrect closed array'

    positions.update(100)
    opened = positions.fetch_opened()
    closed = positions.fetch_closed()
    value = positions.value()

    assert value == 0.25, 'positions did not properly compute value'
    assert len(opened) == 1, 'positions returned incorrect opened array'
    assert len(closed) == 1, 'positions returned incorrect closed array'
