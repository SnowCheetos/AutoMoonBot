from enum import Enum, auto
from typing import Hashable

from backend.data import Element, nodes as n


class Tense(Enum):
    Past = auto()
    Present = auto()
    Future = auto()


class Aspect(Enum):
    Simple = auto()
    Perfect = auto()
    Continuous = auto()
    PerfectContinuous = auto()


class Edge(Element):
    tense = None
    aspect = None
    source_type = None
    target_type = None

    def __init__(
        self,
        source: Hashable,
        target: Hashable,
        on_error: str = "omit",
        **kwargs,
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

    @property
    def tense(self):
        return self.__class__.tense
    
    @property
    def aspect(self):
        return self.__class__.aspect

    @property
    def source_type(self):
        return self.__class__.source_type

    @property
    def target_type(self):
        return self.__class__.target_type


class Issues(Edge):
    tense = Tense.Present
    aspect = Aspect.Simple
    source_type = n.Company
    target_type = n.Equity

    def __init__(
        self,
        source: Hashable,
        target: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            on_error=on_error,
            **kwargs,
        )


class Serves(Edge):
    tense = Tense.Present
    aspect = Aspect.Simple
    source_type = n.Author
    target_type = n.Publisher

    def __init__(
        self,
        source: Hashable,
        target: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            on_error=on_error,
            **kwargs,
        )


class Drafted(Edge):
    tense = Tense.Past
    aspect = Aspect.Simple
    source_type = n.Author
    target_type = n.News

    def __init__(
        self,
        source: Hashable,
        target: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            on_error=on_error,
            **kwargs,
        )


class Published(Edge):
    tense = Tense.Past
    aspect = Aspect.Simple
    source_type = n.Publisher
    target_type = n.News

    def __init__(
        self,
        source: Hashable,
        target: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(
            source=source,
            target=target,
            on_error=on_error,
            **kwargs,
        )


class Referenced(Edge):
    tense = Tense.Past
    aspect = Aspect.Simple
    source_type = n.News
    target_type = n.Equity

    def __init__(
        self,
        source: Hashable,
        target: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(source, target, on_error, **kwargs)


class Affects(Edge):
    tense = Tense.Past
    aspect = Aspect.PerfectContinuous
    source_type = n.News
    target_type = n.Equity

    def __init__(
        self,
        source: Hashable,
        target: Hashable,
        on_error: str = "omit",
        **kwargs,
    ) -> None:
        super().__init__(source, target, on_error, **kwargs)
