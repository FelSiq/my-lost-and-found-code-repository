"""Algorithm to sort a DAG.

The edges may be any real value.

Runs in Theta(V + E) time.
"""
import typing as t

import numpy as np

_adj_list_t = t.Dict[t.Any, t.Tuple[t.Sequence[t.Any], t.Sequence[float]]]


def _topological_sort(graph):
    """Topological sort using dependency count."""
    ALREADY_PROCESSED = -1

    ans = []
    dep_count = dict.fromkeys(graph.keys(), 0)

    for adj_list in graph.values():
        for adj_node, _ in adj_list:
            dep_count[adj_node] += 1

    break_flag = True
    while break_flag:
        break_flag = False
        for node in dep_count:
            if dep_count[node] == 0:
                dep_count[node] = ALREADY_PROCESSED
                ans.append(node)
                for adj_node, _ in graph[node]:
                    dep_count[adj_node] -= 1

                break_flag = True

    if len(ans) == len(graph):
        return ans

    return None


def sort_dag(graph: _adj_list_t, source: t.Any) -> t.Dict[t.Any, t.Tuple[t.Any, float]]:
    res = dict.fromkeys(graph.keys(), (None, np.inf))
    res[source] = None, 0.0

    order = _topological_sort(graph)

    if order is None:
        raise ValueError("Can't perform topological sort. Cycle detected.")

    for u in order:
        for v, w in graph[u]:
            new_w_candidate = w + res[u][1]
            if res[v][1] > new_w_candidate:
                res[v] = u, new_w_candidate

    return res


def _test() -> None:
    np.random.seed(16)

    graph = {
        v_ind: [
            (i, 10 * np.random.random() - 1)
            for i, p_edge in enumerate(np.random.random(15 - v_ind - 1), v_ind + 1)
            if p_edge <= 0.1
        ]
        for v_ind in np.arange(15)
    }

    res = sort_dag(graph, source=0)

    for v, adj in graph.items():
        print(v, "- Sucessor:", res[v][0], f"- Distance from origin: {res[v][1]:.2f}")
        for ew in adj:
            print("   ", ew)


if __name__ == "__main__":
    _test()
