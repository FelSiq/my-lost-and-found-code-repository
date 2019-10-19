r"""Checks is a digraph (directed graph) contains a cycle.

A digraph (directed graph) is acyclic if and only DFS (Depth-First
Search) detects no 'back edges'.

A 'back edge' connects a node to some of its ancestors in the DFS tree.

Running time of this algorithm is $\Theta(|V| + |E|)$, since it
basically fully relies on the DFS algorithm.

This application is an example of how the recursive approach of the
DFS fits so much better than the iterative approach, as we need
information related to when some node is finished in the DFS search.
"""
import typing as t
import enum

import numpy as np


class NodeStatus(enum.IntEnum):
    """Status of nodes during DFS.

    Two status are not sufficient for this application, as we
    need to be able to distinguish explored nodes from previous
    DFS to nodes discovered on the current DFS.
    """
    UNVISITED = 0
    DISCOVERED = 1
    EXPLORED = 2


def _detected_cycle(node_status: t.List[int], ind_adj_node: int) -> bool:
    return node_status[ind_adj_node] == NodeStatus.DISCOVERED


def _dfs(digraph: np.ndarray, node_status: t.List[int],
         ind_cur_node: int) -> bool:
    node_status[ind_cur_node] = NodeStatus.DISCOVERED

    for ind_adj_node, weight in enumerate(digraph[ind_cur_node, :]):
        if weight > 0:
            if _detected_cycle(node_status, ind_adj_node):
                return False

            if node_status[ind_adj_node] == NodeStatus.UNVISITED:
                if not _dfs(digraph, node_status, ind_adj_node):
                    return False

    node_status[ind_cur_node] = NodeStatus.EXPLORED
    return True


def is_acyclic(digraph: np.ndarray) -> bool:
    """Checks if a Digraph is a DAG."""
    num_nodes = digraph.shape[0]
    node_status = np.full(num_nodes, NodeStatus.UNVISITED, dtype=int)
    unvisited_nodes = {i for i in np.arange(num_nodes)}

    while unvisited_nodes:
        random_source_ind = unvisited_nodes.pop()

        if not _dfs(digraph, node_status, random_source_ind):
            return False

        unvisited_nodes -= set(np.where(node_status == NodeStatus.EXPLORED)[0])

    return True


def _test():
    graph = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
    ],
                     dtype=int)

    res = is_acyclic(graph)
    print(res)


if __name__ == "__main__":
    _test()
