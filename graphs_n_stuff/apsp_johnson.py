import typing as t

import numpy as np

import bellman_ford
import dijkstra

_adj_list_t = t.Dict[t.Any, t.Tuple[t.Sequence[t.Any], t.Sequence[float]]]


def johnson(
    graph: _adj_list_t, new_vert_name: t.Any = -1
) -> t.Tuple[np.ndarray, t.Dict[t.Any, int]]:
    if new_vert_name in graph:
        raise ValueError(
            f"'new_vert_name' ({new_vert_name}) in graph. Please use another value."
        )

    # Add a new supersource vertex and run Bellman-Ford algorithm
    graph[new_vert_name] = [(v, 0) for v in graph.keys()]
    bf_res, bf_success = bellman_ford.bellman_ford(graph, source=new_vert_name)
    graph.pop(new_vert_name)

    # Check for negative weight cycles
    if not bf_success:
        raise ValueError("Graph has a negative weight cycle.")

    # Adjust weights to be all non-negative using Bellman-Ford results
    for v, e in graph.items():
        for ind, (v_adj, w) in enumerate(e):
            graph[v][ind] = v_adj, w + bf_res[v][1] - bf_res[v_adj][1]

    num_vert = len(graph.keys())
    D = np.zeros((num_vert, num_vert))
    vert_inds = {v: i for i, v in enumerate(graph.keys())}

    # Run Dijkstra's algorithm for every node as the source
    for v, e in graph.items():
        dj_res = dijkstra.dijkstra(graph, source=v)
        cur_v_ind = vert_inds[v]
        for v_tg, (_, w) in dj_res.items():
            tg_ind = vert_inds[v_tg]
            D[cur_v_ind, tg_ind] = w + bf_res[v_tg][1] - bf_res[v][1]

    # Set graph weights back to original value
    for v, e in graph.items():
        for ind, (v_adj, w) in enumerate(e):
            graph[v][ind] = v_adj, w - bf_res[v][1] + bf_res[v_adj][1]

    return D, vert_inds


def _test() -> None:
    W = np.array(
        [
            [0, 3, 8, np.inf, -4],
            [np.inf, 0, np.inf, 1, 7],
            [np.inf, 4, 0, np.inf, np.inf],
            [2, np.inf, -5, 0, np.inf],
            [np.inf, np.inf, np.inf, 6, 0],
        ]
    )

    graph = {
        key: [(i, w) for i, w in enumerate(W[key, :]) if np.isfinite(w)]
        for key in np.arange(W.shape[0])
    }

    print(graph)

    res = johnson(graph)

    print(res)


if __name__ == "__main__":
    _test()
