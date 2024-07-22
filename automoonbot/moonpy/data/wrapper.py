from moonrs import HeteroGraph


class HeteroGraphWrapper(HeteroGraph):
    def __init__(self) -> None:
        super().__init__()
        self._db = None

    def reset(self):
        pass

    def update(self):
        pass

    def add_node(self):
        pass

    def remove_node(self):
        pass