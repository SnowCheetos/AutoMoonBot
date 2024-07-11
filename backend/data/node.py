from backend.data import Element
from typing import Hashable


class Node(Element):
    def __init__(self, index: Hashable, on_error: str = "omit", **kwargs) -> None:
        super().__init__(on_error=on_error, **kwargs)

        assert isinstance(
            index, Hashable
        ), f"Invalid index type {type(index)}, must be a hashable type"

        self.index = index
