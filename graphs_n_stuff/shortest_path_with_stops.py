"""Find the shortest path between two nodes needing to recharge inbetween.

Each 'recharge' station fully recharges the 'walker' energy. The 'walker' can
only move from a vertex to another if it have at least as energy as the edge
cost.
"""
import typing as t

import numpy as np
import heapq

import apsp_johnson
import dijkstra

_adj_list_t = t.Dict[t.Any, t.Tuple[t.Sequence[t.Any], t.Sequence[float]]]


def spws(
    graph: _adj_list_t,
    max_dist: np.number,
    stops: t.Collection[t.Any],
    source: t.Any,
    target: t.Any,
) -> np.number:
    def _traceback(
        res: t.Dict[t.Any, t.Tuple[t.Any, float]], node_start: t.Any, node_end: t.Any
    ) -> np.number:
        cur_vert, prev_vert = node_end, None

        while cur_vert is not None:
            prev_vert = cur_vert
            cur_vert, cur_dist = res[cur_vert]

        if prev_vert != node_start:
            return -1

        dist = res[node_end][1]

        return dist

    D, vert_inds = apsp_johnson.johnson(graph)

    new_graph = {}

    stops = stops.union({source, target})

    for v in stops:
        new_graph[v] = []
        for u in stops:
            dist = D[vert_inds[v], vert_inds[u]]
            if v != u and dist <= max_dist:
                new_graph[v].append((u, dist))

    res = dijkstra.dijkstra(new_graph, source=source)

    return _traceback(res, source, target)


def _test() -> None:
    graph = {
        "New york": [("A", 70), ("B", 50)],
        "A": [("New york", 70), ("B", 10), ("C", 60)],
        "B": [("New york", 50), ("C", 50), ("A", 10)],
        "C": [("B", 50), ("D", 10), ("F", 60)],
        "D": [("A", 60), ("C", 10), ("E", 50)],
        "F": [("C", 60), ("E", 10), ("Boston", 80)],
        "E": [("Boston", 60), ("D", 50), ("F", 10)],
        "Boston": [("E", 60), ("F", 80)],
    }

    res = spws(
        graph,
        max_dist=100,
        stops={"New york", "A", "D", "C", "F", "Boston"},
        source="Boston",
        target="New york",
    )

    print(res)


if __name__ == "__main__":
    _test()
