import pytest
import torch
from backend.data import Element


def test_basics():
    element = Element(on_error="omit")
    assert element, "Failed to initialize element with omit on-error"

    element = Element(on_error="raise")
    assert element, "Failed to initialize element with raise on-error"

    with pytest.raises(Exception) as e:
        element = Element(on_error="invalid")
    assert isinstance(e.value, AssertionError), "Failed to check for on-error type"

    element = Element(var=1)
    assert element.var == 1, "Incorrect instantiation of element attr var"
    
    with pytest.raises(Exception) as e:
        _ = element.x
    assert isinstance(e.value, AttributeError), "Failed to check for attribute existence"

def test_attr():
    element = Element(on_error="omit")
    assert element.attr() == None, "Element failed to omit None"

    class Example(Element):
        def __init__(self, on_error, **kwargs):
            super().__init__(on_error, **kwargs)
        
        def get_attr(self):
            return {"var": self.var}
    
    element = Example(on_error="omit", var=1)
    assert element.attr() == {"var": 1}, "Element failed to use `get_attr(...)`"

    # TODO Lambda scripts

def test_tensor():
    element = Element(on_error="omit")
    assert element.tensor() == None, "Element failed to omit None"

    class Example(Element):
        def __init__(self, on_error, **kwargs):
            super().__init__(on_error, **kwargs)
        
        def get_tensor(self):
            return torch.tensor(self.var)
    
    element = Example(on_error="omit", var=1)
    assert element.tensor().item() == 1, "Element failed to use `get_tensor(...)`"

    # TODO Lambda scripts

def test_update():
    element = Element(on_error="omit")
    assert element.update() == False, "Element failed to return False"

    class Example(Element):
        def __init__(self, on_error, **kwargs):
            super().__init__(on_error, **kwargs)
        
        def get_update(self, **kwargs):
            return kwargs
    
    element = Example(on_error="omit", var=1)
    assert element.update(var=1) == True, "Element failed to perform update"
    assert element.var == 1, "Element updated the wrong value"

    # TODO Lambda scripts