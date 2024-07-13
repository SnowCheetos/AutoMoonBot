import pytest
import torch
from automoonbot.data import Element


def test_basics():
    element = Element(on_error="omit")
    assert element, "Failed to initialize element with omit on-error"

    element = Element(on_error="raise")
    assert element, "Failed to initialize element with raise on-error"

    with pytest.raises(ValueError):
        element = Element(on_error="invalid")

    element = Element(var=1)
    assert element.var == 1, "Incorrect instantiation of element attr var"

    with pytest.raises(AttributeError):
        _ = element.x


def test_attr():
    element = Element(on_error="omit")
    assert element.attr == None, "Element failed to omit None"

    class Example(Element):
        def __init__(self, on_error, **kwargs):
            super().__init__(
                mutable=True,
                on_error=on_error,
                **kwargs,
            )

        def get_attr(self):
            return {"var": self.var}

    element = Example(on_error="omit", var=1)
    assert element.attr == {"var": 1}, "Element failed to use `get_attr(...)`"

    # TODO Lambda scripts


def test_tensor():
    element = Element(on_error="omit")
    assert element.tensor == None, "Element failed to omit None"

    class Example(Element):
        def __init__(self, on_error, **kwargs):
            super().__init__(
                mutable=True,
                on_error=on_error,
                **kwargs,
            )

        def get_tensor(self):
            return torch.tensor(self.var)

    element = Example(on_error="omit", var=1)
    assert element.tensor.item() == 1, "Element failed to use `get_tensor(...)`"

    # TODO Lambda scripts


def test_update():
    element = Element(on_error="omit")
    assert element.update() == False, "Element failed to return False"

    class Example(Element):
        def __init__(self, on_error, **kwargs):
            super().__init__(
                mutable=True,
                on_error=on_error,
                **kwargs,
            )

        def get_update(self, **kwargs):
            return kwargs

    element = Example(on_error="omit", var=1)
    assert element.update(var=1) == True, "Element failed to perform update"
    assert element.var == 1, "Element updated the wrong value"

    class Example(Element):
        def __init__(self, on_error, **kwargs):
            super().__init__(
                mutable=False,
                on_error=on_error,
                **kwargs,
            )

        def get_update(self, **kwargs):
            return kwargs

    element = Example(on_error="raise", var=1)
    with pytest.raises(AttributeError):
        element.update(var=2)

    # TODO Lambda scripts
