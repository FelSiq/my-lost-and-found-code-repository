"""Dijkstra's Algorithm for single-source shortest paths.

Runs in O(E + VlgV) in implemented with a Fibonacci Heap.

Works only if all graph edges have non-negative weights.
"""
import typing as t

import numpy as np

import heapq

_adj_list_t = t.Dict[t.Any, t.Tuple[t.Sequence[t.Any], t.Sequence[float]]]


def dijkstra(graph: _adj_list_t, source: t.Any) -> t.Dict[t.Any, t.Tuple[t.Any, float]]:
    """Dijkstra algorithm using a 'simple' heap.

    Later I should reimplement it using a Fibonacci Heap.
    """
    heap = [(0, source)]
    res = {v: (None, np.inf) for v in graph.keys()}
    res[source] = None, 0.0

    visited = set()

    while heap:
        cur_dist, cur_v = heapq.heappop(heap)

        if cur_v in visited:
            continue

        visited.add(cur_v)

        for v_adj, w in graph[cur_v]:
            new_w_candidate = cur_dist + w
            if res[v_adj][1] > new_w_candidate:
                res[v_adj] = cur_v, new_w_candidate
                heapq.heappush(heap, (new_w_candidate, v_adj))

    return res


def _test() -> None:
    np.random.seed(16)

    graph = {
        v_ind: [
            (i, 10 * np.random.random())
            for i, p_edge in enumerate(np.random.random(15))
            if p_edge <= 0.1
        ]
        for v_ind in np.arange(15)
    }

    res = dijkstra(graph, source=0)

    for v, adj in graph.items():
        print(v, "- Sucessor:", res[v][0], f"- Distance from origin: {res[v][1]:.2f}")
        for ew in adj:
            print("   ", ew)


if __name__ == "__main__":
    _test()
