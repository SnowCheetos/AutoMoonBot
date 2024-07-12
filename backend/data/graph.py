import networkx as nx

from typing import Set
from backend.data import nodes as n, edges as e


class HeteroGraph(nx.MultiDiGraph):
    def __init__(
        self,
    ):
        super().__init__()

        self._node_memo = {}
        self._edge_memo = {}
        self._edge_element_memo = {}

    def clear(self) -> None:
        self._node_memo = {}
        self._edge_memo = {}
        self._edge_element_memo = {}
        super().clear()

    def add_node(self, element: n.Node, index: str, **kwargs) -> None:
        if self.has_node(index):
            return
        node = element(index=index, **kwargs)
        super().add_node(index, element=node)
        if element not in self._node_memo.keys():
            self._node_memo[element] = {index}
        else:
            self._node_memo[element].add(index)

    def remove_node(self, index: str) -> None:
        if not self.has_node(index):
            return
        super().remove_node(index)

    def add_edge(self, element: e.Edge, src: str, tgt: str, **kwargs) -> None:
        edge = element(src, tgt, **kwargs)
        if edge.attr == None:
            return
        super().add_edge(src, tgt, key=edge.name, element=edge)
        if element not in self._edge_memo.keys():
            self._edge_memo[element] = {(src, tgt)}
        else:
            self._edge_memo[element].add((src, tgt))

    def remove_edge(self, src: str, tgt: str, edge: e.Edge) -> None:
        if not self.has_edge(src, tgt, key=edge.name):
            return
        return super().remove_edge(src, tgt, key=edge.name)

    def compute_edges(self, clear: bool=False) -> None:
        if len(self.nodes) <= 1:
            return
        
        if clear:
            self.clear_edges()

        for node in self.nodes:
            self.compute_node_edges(node)

    def compute_node_edges(self, index: str) -> None:
        if not self.has_node(index):
            return
        
        for node in self.nodes:
            if node != index:
                self.add_edges(index, node)

    def compute_edge_elements(self, src: n.Node, tgt: n.Node) -> Set[e.Edge]:
        key = (src, tgt)
        memo = self._edge_element_memo.get(key, None)
        if memo:
            return memo

        self._edge_element_memo[key] = set()
        for edge in e.Edges:
            if edge.source_type == src and edge.target_type == tgt:
                self._edge_element_memo[key].add(edge)
        return self._edge_element_memo[key]

    def add_edges(self, src: str, tgt: str) -> None:
        nodes = dict(self.nodes(data="element"))
        source = nodes.get(src, None)
        target = nodes.get(tgt, None)
        if not (source or target):
            return
        edges = self.compute_edge_elements(source, target)
        for edge in edges:
            self.add_edge(edge, src, tgt, src_element=source, tgt_element=target)