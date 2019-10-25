"""Find the Minimum Spanning Tree (MST) of a undirected Graph using Kruskal algorithm."""
import typing as t

import numpy as np

import graph_disjoint_sets


def mst_kruskal(
        graph: np.ndarray,
        return_total_weight: bool = False,
) -> t.Union[np.ndarray, t.Tuple[np.ndarray, bool]]:
    """Find the Minimum Spanning Tree (MST) using Kruskal algorithm.

    Notes
    -----
    Uses a set-disjoint rooted tree data structure to store the
    unconnencted components (trees) of the ``kruskal forest``.
    """
    model = graph_disjoint_sets.GraphDisjointSets()

    num_nodes = graph.shape[0]

    for ind_node in np.arange(num_nodes):
        model.make_set(ind_node)

    mst = np.zeros(graph.shape)

    # Note: graph is undirected, therefore only a tringle of
    # the adjacency matrix must be considered.
    edges_coord = np.where(np.triu(graph, k=1) > 0)
    edges_sorted = sorted(
        zip(graph[edges_coord], *edges_coord), key=lambda item: item[0])

    total_weight = 0
    for weight, edge_a, edge_b in edges_sorted:
        if not model.same_component(edge_a, edge_b):
            mst[edge_a, edge_b] = mst[edge_b, edge_a] = weight
            model.union(edge_a, edge_b)
            total_weight += weight

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

    mst, total_weight = mst_kruskal(graph, return_total_weight=True)

    print("Total weight:", total_weight)
    print(mst)


if __name__ == "__main__":
    _test()
