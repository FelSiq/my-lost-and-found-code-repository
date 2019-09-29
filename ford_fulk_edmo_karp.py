"""Edmonds-Karp implementation of Ford-Fulkerson method for flow networks.

Finds the maximum network flow in O(V * E**2) time complexity.
"""
import typing as t
import warnings

import numpy as np


def _traceback(graph: np.ndarray, predecessor_vec: np.ndarray, id_root: int,
               id_target: int) -> t.Tuple[np.ndarray, t.Union[int, float]]:
    """Use the predecessor vector to build the found path."""
    ans = [id_target]
    id_cur_node = id_target
    bottleneck_val = np.inf

    while id_cur_node != id_root:
        prev_node = predecessor_vec[id_cur_node]
        bottleneck_val = min(bottleneck_val, graph[prev_node, id_cur_node])
        id_cur_node = prev_node
        ans.insert(0, id_cur_node)

    return np.array(ans), bottleneck_val


def _bfs(graph: np.ndarray, id_root: int, id_target: int
         ) -> t.Tuple[t.Optional[np.ndarray], t.Union[int, float]]:
    """Breadth-first search from node ``id_root`` to node ``id_target``."""
    queue = [id_root]

    predecessor_vec = np.full(graph.shape[0], -1)
    predecessor_vec[id_root] = -2  # Another invalid value

    while queue:
        id_cur_node = queue.pop()

        if id_cur_node == id_target:
            return _traceback(graph, predecessor_vec, id_root, id_target)

        for id_adj_node, edge_weight in enumerate(graph[id_cur_node]):
            if edge_weight > 0 and predecessor_vec[id_adj_node] == -1:
                predecessor_vec[id_adj_node] = id_cur_node
                queue.insert(0, id_adj_node)

    return None, 0


def _check_self_loops(graph: np.ndarray,
                      inplace: bool = False,
                      verbose: bool = False) -> np.ndarray:
    """Check if ``graph`` has self loops and remove then if necessary."""
    _removed_loops_count = 0

    if inplace:
        new_graph = graph

    else:
        new_graph = graph.copy()

    if verbose:
        for i in np.arange(new_graph.shape[0]):
            if new_graph[i, i]:
                _removed_loops_count += 1
                new_graph[i, i] -= new_graph[i, i]

        print(
            "Removed {} self-loops in new_graph.".format(_removed_loops_count))

    else:
        for i in np.arange(new_graph.shape[0]):
            new_graph[i, i] -= new_graph[i, i]

    return new_graph


def _remove_antiparallel_edges(graph: np.ndarray,
                               verbose: bool = False) -> np.ndarray:
    """Workaround for antiparallel edges (u, v) and (v, u).

    The strategy adopted is to create a new node w such as the edge (u, v)
    is replaced by $e_{1} = (u, w)$ and $e_{2} = (w, v)$.

    Both $e_{1}$ and $e_{2}$ has the same capacity of (u, v).
    """
    new_nodes = []

    num_node = graph.shape[0]

    for node_id_a in np.arange(1, num_node):
        for node_id_b in np.arange(0, node_id_a):
            if graph[node_id_a, node_id_b] > 0 and graph[node_id_b,
                                                         node_id_a] > 0:
                new_nodes.append((node_id_a, node_id_b))

    new_graph = np.zeros((num_node + len(new_nodes),
                          num_node + len(new_nodes)))
    new_graph[:num_node, :num_node] = graph

    for new_node_shift_val, adj_nodes in enumerate(new_nodes):
        new_node_id = num_node + new_node_shift_val
        adj_node_a, adj_node_b = adj_nodes

        new_graph[adj_node_a, adj_node_b] = 0
        new_graph[adj_node_a, new_node_id] = graph[adj_node_a, adj_node_b]
        new_graph[new_node_id, adj_node_b] = graph[adj_node_a, adj_node_b]

    if verbose:
        print("Removed {} antiparallel edges = total of new vertices "
              "added.".format(len(new_nodes)))

    return new_graph


def edkarp_maxflow(graph: np.ndarray,
                   id_source: int,
                   id_sink: int,
                   check_antiparallel_edges: bool = True,
                   check_self_loops: bool = True,
                   inplace: bool = False,
                   verbose: bool = False) -> t.Union[int, float]:
    """."""
    if inplace and check_antiparallel_edges:
        inplace = False
        warnings.warn(
            "Can't make changes in-place with 'check_antiparallel_edges' "
            "activated. Disabling in-place changes.", UserWarning)

    if check_self_loops:
        graph = _check_self_loops(graph, inplace=inplace, verbose=verbose)

    if check_antiparallel_edges:
        graph = _remove_antiparallel_edges(graph, verbose=verbose)

    graph_residual = graph.copy()

    path, bottleneck_val = _bfs(
        graph_residual, id_root=id_source, id_target=id_sink)

    while path is not None:
        for node_id_a, node_id_b in zip(path[:-1], path[1:]):
            if graph[node_id_a, node_id_b] > 0:
                graph_residual[node_id_a, node_id_b] -= bottleneck_val

            else:
                graph_residual[node_id_a, node_id_b] += bottleneck_val

        path, bottleneck_val = _bfs(
            graph_residual, id_root=id_source, id_target=id_sink)

    max_flow = np.sum(graph[id_source, :] - graph_residual[id_source, :])

    if verbose:
        print("Final residual network:")
        print(graph_residual)

        print("Flow:")
        print(graph - graph_residual)

    return max_flow


if __name__ == "__main__":
    GRAPH = np.array([
        [0, 16, 13, 0, 0, 0],
        [0, 0, 0, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0],
    ])
    MAX_FLOW = edkarp_maxflow(GRAPH, 0, 5, verbose=True)
    print("Max flow:", MAX_FLOW)
