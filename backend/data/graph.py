import networkx as nx

from typing import List, Set, Tuple
from backend.data import nodes as n, edges as e


class HeteroGraph(nx.MultiDiGraph):
    def __init__(
        self,
    ):
        super().__init__()

        self._node_memo = dict()
        self._edge_memo = dict()
        self._edge_element_memo = dict()

    def clear(self) -> None:
        self._node_memo.clear()
        self._edge_memo.clear()
        self._edge_element_memo.clear()
        super().clear()

    def add_node(self, element: n.Node, index: str, **kwargs) -> None:
        if self.has_node(index):
            return
        node = element(index=index, **kwargs)
        super().add_node(index, element=node)
        if element not in self._node_memo.keys():
            self._node_memo[element] = set({index})
        else:
            self._node_memo[element].add(index)

    def get_node(self, index: str) -> n.Node | None:
        return dict(self.nodes(data="element")).get(index, None)

    def get_nodes(self, element: n.Node) -> Set[str]:
        return self._node_memo.get(element)

    def remove_node(self, index: str) -> None:
        if not self.has_node(index):
            return
        element = self.get_node(index)
        self._node_memo[element].pop(index)
        super().remove_node(index)

    def add_edge(self, element: e.Edge, src: str, tgt: str, **kwargs) -> None:
        edge = element(src, tgt, **kwargs)
        if edge.attr == None:
            return
        super().add_edge(src, tgt, key=edge.name, element=edge)
        if element not in self._edge_memo.keys():
            self._edge_memo[element] = set({(src, tgt)})
        else:
            self._edge_memo[element].add((src, tgt))

    def get_edges_uv(self, edge: e.Edge) -> Set[Tuple[str]] | None:
        return self._edge_memo.get(edge, None)

    def get_edges_element(self, edge: e.Edge) -> List[e.Edge] | None:
        uv = self.get_edges_uv(edge)
        if not uv:
            return None
        edges = []
        for u, v in uv:
            element = self.get_edge_data(u, v, key=edge.name).get("element")
            edges.append(element)
        return edges

    def remove_edge(self, src: str, tgt: str, edge: e.Edge) -> None:
        if not self.has_edge(src, tgt, key=edge.name):
            return
        return super().remove_edge(src, tgt, key=edge.name)

    def compute_edges(self, clear: bool = False) -> None:
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
        source = self.get_node(src)
        target = self.get_node(tgt)
        if not (source or target):
            return
        edges = self.compute_edge_elements(source, target)
        for edge in edges:
            self.add_edge(edge, src, tgt, src_element=source, tgt_element=target)
