from backend.data import Element
from typing import Hashable


class Edge(Element):
    def __init__(
        self, source: Hashable, target: Hashable, on_error: str = "omit", **kwargs
    ) -> None:
        super().__init__(on_error=on_error, **kwargs)

        assert isinstance(
            source, Hashable
        ), f"Invalid source type {type(source)}, must be a hashable type"

        assert isinstance(
            target, Hashable
        ), f"Invalid target type {type(target)}, must be hashable type"

        self.source = source
        self.target = target
