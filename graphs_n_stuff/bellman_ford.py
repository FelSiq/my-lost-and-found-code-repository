"""Bellman-Ford algorithm.

Solves a general single-source shortest path in O(VE) time complexity.
"""
import typing as t

import numpy as np

_adj_list_t = t.Dict[t.Any, t.Tuple[t.Sequence[t.Any], t.Sequence[float]]]


def bellman_ford(
    graph: _adj_list_t, source: t.Any
) -> t.Tuple[t.Dict[t.Any, t.Tuple[t.Any, float]], bool]:
    res = dict.fromkeys(graph.keys(), (None, np.inf))
    res[source] = None, 0.0

    n = len(graph.keys())

    # Perform at most n - 1 relaxations in each vertex
    for _ in np.arange(n - 1):
        for v, adj in graph.items():
            for v_adj, w in adj:
                # Apply relaxation if needed
                new_w_candidate = res[v][1] + w
                if res[v_adj][1] > new_w_candidate:
                    res[v_adj] = v, new_w_candidate

    # Check for negative-weight cycles and return False if and only if there
    # exists such a cycle.
    for v, adj in graph.items():
        for v_adj, w in adj:
            if res[v_adj][1] > res[v][1] + w:
                return res, False

    return res, True


def _test() -> None:
    np.random.seed(16)

    graph = {
        v_ind: [
            (i, 10 * np.random.random() - 1)
            for i, p_edge in enumerate(np.random.random(15))
            if p_edge <= 0.1
        ]
        for v_ind in np.arange(15)
    }

    res, success = bellman_ford(graph, source=0)

    for v, adj in graph.items():
        print(v, "- Sucessor:", res[v][0], f"- Distance from origin: {res[v][1]:.2f}")
        for ew in adj:
            print("   ", ew)

    if not success:
        print("Algorithm failed since there exists a negative weight cycle.")


if __name__ == "__main__":
    _test()
