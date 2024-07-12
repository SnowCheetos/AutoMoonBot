from enum import Enum, auto
from typing import Hashable, Set

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
    """
    e.g. Company A issues stock A
    """

    name = "issues"
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
            mutable=False,
            on_error=on_error,
            **kwargs,
        )

    # TODO This ugly mess made me realized I wasn't crazy over the need for a custom script
    def get_attr(self):
        if not (self.src_element or self.tgt_element):
            return None
        elif not (self.src_element.symbol or self.tgt_element.symbol):
            return None
        elif self.src_element.symbol != self.tgt_element.symbol:
            return None
        else:
            return {0: 1}

    def get_tensor(self):
        pass


class Drafted(Edge):
    """
    e.g. Author drafted news at 4pm on June 16th
    """

    name = "drafted"
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
            mutable=False,
            on_error=on_error,
            **kwargs,
        )

    def get_attr(self):
        pass

    def get_tensor(self):
        pass


class Published(Edge):
    """
    e.g. Publisher published news at 9:45am
    """

    name = "published"
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
            mutable=False,
            on_error=on_error,
            **kwargs,
        )

    def get_attr(self):
        pass

    def get_tensor(self):
        pass


class Serves(Edge):
    """
    e.g. Author have been working at publisher since 2012
    """

    name = "serves"
    tense = Tense.Present
    aspect = Aspect.Perfect
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


class Employs(Edge):
    """
    e.g. Publisher pays author $80,000 per year
    """

    name = "employs"
    tense = Tense.Past
    aspect = Aspect.Simple
    source_type = n.Publisher
    target_type = n.Author

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


class Referenced(Edge):
    """
    e.g. News referenced stock yesterday
    """

    name = "referenced"
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
        super().__init__(
            source=source,
            target=target,
            mutable=False,
            on_error=on_error,
            **kwargs,
        )

    def get_attr(self):
        pass

    def get_tensor(self):
        pass


class Moves(Edge):
    """
    e.g. Stock has been volatile since news published
    """

    name = "moves"
    tense = Tense.Present
    aspect = Aspect.Perfect
    source_type = n.News
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


Edges: Set[Edge] = {
    Issues,
    Drafted,
    Published,
    Serves,
    Employs,
    Referenced,
    Moves,
}
