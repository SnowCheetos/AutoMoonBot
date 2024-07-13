import torch
from functools import lru_cache
from typing import Hashable, Set

from backend.data import Element


class Nodes(type):
    subclasses = set()

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        if bases != (Element,):
            cls.subclasses.add(new_class)
        return new_class

    @classmethod
    def get(cls) -> Set[Element]:
        return cls.subclasses


class Node(Element, metaclass=Nodes):
    tensor_dim: int = None

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

        self.index = index

    def __init_subclass__(cls, *args, **kwargs) -> None:
        super().__init_subclass__(*args, **kwargs)
        cls.name = cls.__name__.lower()

    @property
    @lru_cache(maxsize=None)
    def tensor_dim(self) -> int:
        return self.__class__.tensor_dim


class Company(Node):
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
