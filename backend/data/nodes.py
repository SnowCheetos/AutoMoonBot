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
    name = "company"

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

    def get_attr(self):
        pass

    def get_tensor(self):
        pass

    def get_update(self):
        pass


class Equity(Node):
    name = "equity"

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

    def get_attr(self):
        pass

    def get_tensor(self):
        pass

    def get_update(self):
        pass


class News(Node):
    name = "news"

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

    def get_attr(self):
        pass

    def get_tensor(self):
        pass


class Author(Node):
    name = "author"

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

    def get_attr(self):
        pass

    def get_tensor(self):
        pass

    def get_update(self):
        pass


class Publisher(Node):
    name = "publisher"

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

    def get_attr(self):
        pass

    def get_tensor(self):
        pass

    def get_update(self):
        pass


class Topic(Node):
    name = "topic"

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

    def get_attr(self):
        pass

    def get_tensor(self):
        pass
