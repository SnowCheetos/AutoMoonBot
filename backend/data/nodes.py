from enum import Enum
from typing import Hashable

from backend.data import Element


class Node(Element):
    def __init__(
        self,
        index: Hashable,
        mutable: bool = True,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            mutable=mutable,
            on_error=on_error,
            **kwargs,
        )

        assert isinstance(
            index, Hashable
        ), f"Invalid index type {type(index)}, must be a hashable type"

        self.index = index


class Company(Node):
    def __init__(
        self,
        index: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            index=index,
            mutable=True,
            on_error=on_error,
            **kwargs,
        )


class Equity(Node):
    def __init__(
        self,
        index: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            index=index,
            mutable=True,
            on_error=on_error,
            **kwargs,
        )


class News(Node):
    def __init__(
        self,
        index: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            index=index,
            mutable=False,
            on_error=on_error,
            **kwargs,
        )


class Author(Node):
    def __init__(
        self,
        index: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            index=index,
            mutable=True,
            on_error=on_error,
            **kwargs,
        )


class Publisher(Node):
    def __init__(
        self,
        index: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            index=index,
            mutable=True,
            on_error=on_error,
            **kwargs,
        )


class Topic(Node):
    def __init__(
        self,
        index: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            index=index,
            mutable=False,
            on_error=on_error,
            **kwargs,
        )
