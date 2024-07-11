import networkx as nx

from typing import Set, Type
from backend.data import nodes as n, edges as e


class HeteroGraph(nx.MultiDiGraph):
    def __init__(
        self,
    ):
        super().__init__()

        self._node_memo = {}
        self._edge_memo = {}
        self._edge_element_memo = {}

    def add_node(self, element: n.Node, index: str, **kwargs) -> None:
        if self.has_node(index):
            return
        node = element(index=index, **kwargs)
        super().add_node(index, element=node)
        if element not in self._node_memo.keys():
            self._node_memo[element] = {index}
        else:
            self._node_memo[element].add(index)

    def add_edge(self, element: e.Edge, src: str, tgt: str, **kwargs) -> None:
        edge = element(src, tgt, **kwargs)
        super().add_edge(src, tgt, element=edge)
        if element not in self._edge_memo.keys():
            self._edge_memo[element] = {(src, tgt)}
        else:
            self._edge_memo[element].add((src, tgt))

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

    def add_all_edges(self, src: str, tgt: str) -> int:
        nodes = self.nodes(data=True)
        source = nodes.get(src)
        target = nodes.get(tgt)
        edges = self.compute_edge_elements(source, target)
        for edge in edges:
            self.add_edge(edge, src, tgt, src_element=source, tgt_element=target)
        return len(edges)
