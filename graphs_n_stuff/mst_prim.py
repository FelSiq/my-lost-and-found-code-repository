"""Find the Minimum Spanning Tree (MST) of a Graph using Prim algorithm.

Prim algorithm is a Greedy algorithm to finding the MST of a undirected
graph. It works similar to the Djikstra's algorithm for shortests paths
among a pair of nodes.

Each iteration, the Prim algorithm find the minimum weight edge to connect
to the current ``prim tree``, which is a subset of some minimum spanning
tree of the graph. In other words, for any given ith iteration of the Prim
algorithm, the current MST graph is a forest that has (|V| - i) trees.
It means that, in the initialization, the MST graph is a forest with |V|
trees (one for each vertex of the original undirected graph), and in each
iteration a different (single-node) tree is merged to the ``prim tree``,
which consists of the largest tree in the forest. The criteria to chosen
the next tree to be merged follows the greedy approach: the edge with the
smallest weight which connects the current ``prim tree`` and some other
tree is added to the current ``prim tree``, therefore connecting both trees.

The algorithm terminates when there is a single tree in the forest, which
is some minimum spanning tree of the given graph.
"""
import typing as t
import heapq

import numpy as np


def _traceback(predecessor: np.ndarray, graph: np.ndarray) -> np.ndarray:
    """Build the Minimum Spanning Tree matrix from the predecessor information."""
    mst = np.zeros(graph.shape)

    for ind_node, ind_pred in enumerate(predecessor):
        if ind_pred >= 0:
            mst[ind_node, ind_pred] = mst[ind_pred, ind_node] = graph[ind_pred, ind_node]

    return mst


def mst_prim(graph: np.ndarray,
             ind_root: t.Optional[int] = None,
             return_total_weight: bool = False,
             random_state: t.Optional[int] = None,
) -> t.Union[np.ndarray, t.Tuple[np.ndarray, bool]]:
    """Find the Minimum Spanning Tree (MST) using Prim algorithm."""
    num_nodes = graph.shape[0]

    if num_nodes != graph.shape[1]:
        raise ValueError("Graph adjacency matrix must have square "
                         "shape (got {}.)".format(graph.shape))

    if ind_root is None:
        if random_state is not None:
            np.random.seed(random_state)

        ind_root = np.random.randint(num_nodes)
        
    min_pqueue = [(0, ind_root, -2)]
    total_weight = 0

    predecessor = np.full(num_nodes, -1, dtype=int)

    i = 0
    while min_pqueue:
        cur_weight, ind_cur_node, ind_predecessor = heapq.heappop(min_pqueue)

        if predecessor[ind_cur_node] != -1:
            continue

        predecessor[ind_cur_node] = ind_predecessor
        total_weight += cur_weight

        for ind_adj_node, weight in enumerate(graph[ind_cur_node, :]):
            if weight > 0 and predecessor[ind_adj_node] == -1:
                heapq.heappush(min_pqueue, (weight, ind_adj_node, ind_cur_node))

    mst = _traceback(predecessor, graph)

    if return_total_weight:
        return mst, total_weight

    return mst


def _test() -> None:
    graph = np.array([
        [0, 4, 0, 0, 0, 0, 0, 8, 0],
        [0, 0, 8, 0, 0, 0, 0, 11, 0],
        [0, 0, 0, 7, 0, 4, 0, 0, 2],
        [0, 0, 0, 0, 9, 14, 0, 0, 0],
        [0, 0, 0, 0, 0, 10, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 6],
        [0, 0, 0, 0, 0, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    graph += graph.T

    mst, total_weight = mst_prim(graph, return_total_weight=True)

    print("Total weight:", total_weight)
    print(mst)


if __name__ == "__main__":
    _test()
