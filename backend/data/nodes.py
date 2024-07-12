import torch
from typing import Hashable, Set

from backend.data import Element


class Node(Element):
    tensor_dim = None

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

    @property
    def tensor_dim(self) -> int:
        return self.__class__.tensor_dim


class Company(Node):
    name = "company"
    tensor_dim = 10  # Placeholder

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
        return torch.rand(self.tensor_dim, dtype=torch.float)

    def get_update(self):
        pass


class Equity(Node):
    name = "equity"
    tensor_dim = 10  # Placeholder

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
        return torch.rand(self.tensor_dim, dtype=torch.float)

    def get_update(self):
        pass


class News(Node):
    name = "news"
    tensor_dim = 10  # Placeholder

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
        return torch.rand(self.tensor_dim, dtype=torch.float)


class Author(Node):
    name = "author"
    tensor_dim = 10  # Placeholder

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
        return torch.rand(self.tensor_dim, dtype=torch.float)

    def get_update(self):
        pass


class Publisher(Node):
    name = "publisher"
    tensor_dim = 10  # Placeholder

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
        return torch.rand(self.tensor_dim, dtype=torch.float)

    def get_update(self):
        pass


class Topic(Node):
    name = "topic"
    tensor_dim = 10  # Placeholder

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
        return torch.rand(self.tensor_dim, dtype=torch.float)


class Position(Node):
    name = "position"
    tensor_dim = 10  # Placeholder

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
        return torch.rand(self.tensor_dim, dtype=torch.float)


Nodes: Set[Node] = {
    Company,
    Equity,
    News,
    Author,
    Publisher,
    Topic,
    Position,
}
