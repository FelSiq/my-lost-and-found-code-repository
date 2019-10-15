"""Edmonds-Karp implementation of Ford-Fulkerson method for flow networks.

Finds the maximum network flow in O(V * E**2) time complexity.
"""
import typing as t

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


def _check_self_loops(graph: np.ndarray, verbose: bool = False) -> np.ndarray:
    """Check if ``graph`` has self loops and remove then if necessary."""
    _removed_loops_count = 0

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


def _add_supervertex(graph: np.ndarray,
                     id_source: t.Union[int, np.ndarray],
                     id_sink: t.Union[int, np.ndarray],
                     add_source: bool,
                     add_sink: bool,
                     verbose: bool = False) -> t.Tuple[np.ndarray, int, int]:
    """Add a new source/sink node to replace all sources in ``graph``.

    Returns new graph adjacency matrix and the new source/sink node id.
    """
    num_node = graph.shape[0]
    new_graph_dim = num_node + add_source + add_sink
    new_graph = np.zeros((new_graph_dim, new_graph_dim))
    new_graph[:num_node, :num_node] = graph

    new_id_source, new_id_sink = id_source, id_sink

    if add_source:
        new_id_source = num_node
        new_graph[new_id_source, id_source] = np.full(
            id_source.size, graph[id_source, :].max())

        if verbose:
            print("Added supersource (total of {} new edges.)".format(
                id_source.size))

    if add_sink:
        new_id_sink = num_node + add_source
        new_graph[id_sink, new_id_sink] = np.full(id_sink.size,
                                                  graph[:, id_sink].max())

        if verbose:
            print("Added supersink (total of {} new edges.)".format(
                id_sink.size))

    return new_graph, new_id_source, new_id_sink


def extend_vertices_capacity(graph: np.ndarray,
                             vertices_cost: np.ndarray) -> np.ndarray:
    r"""Extend the given ``graph`` to add vertices capacities.

    The strategy adopted is to transform every vertice $v_{i}$
    in two new vertices $v_{i, in}$ and $v_{i, out}$ such that
    $graph[v_{i, in}, v_{i, out}] = \text{vertices\_cost}[i]$.

    All in-edges of $v_{i}$ will be incident to $v_{i, in}$ and
    all out-edges will be incident to $v_{i, out}$.

    The resultant adjacency matrix has size (2n)**2 = 4n**2,
    where n is the number of vertices in the given graph.
    """
    num_vert = graph.shape[0]
    new_graph = np.zeros((2 * num_vert, 2 * num_vert))

    side_diag_size = 2 * num_vert
    connect_related = np.arange(0, side_diag_size,
                                2), np.arange(0, side_diag_size, 2) + 1
    new_graph[connect_related] = vertices_cost

    vals = np.arange(0, 2 * num_vert, 2)
    X, Y = np.meshgrid(vals, vals)

    new_graph[Y + 1, X] = graph

    return new_graph


def edkarp_maxflow(graph: np.ndarray,
                   id_source: t.Union[int, np.ndarray],
                   id_sink: t.Union[int, np.ndarray],
                   check_antiparallel_edges: bool = True,
                   check_self_loops: bool = True,
                   verbose: bool = False) -> t.Union[int, float]:
    """."""
    add_supersource = not isinstance(id_source, int)
    add_supersink = not isinstance(id_sink, int)

    if add_supersource or add_supersink:
        graph, id_source, id_sink = _add_supervertex(
            graph,
            id_source=id_source,
            id_sink=id_sink,
            add_source=add_supersource,
            add_sink=add_supersink,
            verbose=verbose)

    if check_self_loops:
        graph = _check_self_loops(graph, verbose=verbose)

    if check_antiparallel_edges:
        graph = _remove_antiparallel_edges(graph, verbose=verbose)

    graph_residual = graph.copy()

    path, bottleneck_val = _bfs(
        graph_residual, id_root=id_source, id_target=id_sink)

    path_lens = []

    while path is not None:
        path_lens.append(path.size)

        for node_id_a, node_id_b in zip(path[:-1], path[1:]):
            if graph[node_id_a, node_id_b] > 0:
                graph_residual[node_id_a, node_id_b] -= bottleneck_val

            else:
                graph_residual[node_id_a, node_id_b] += bottleneck_val

        path, bottleneck_val = _bfs(
            graph_residual, id_root=id_source, id_target=id_sink)

    max_flow = np.sum(graph[id_source, :] - graph_residual[id_source, :])

    if verbose:
        import matplotlib.pyplot as plt
        print("Final residual network:")
        print(graph_residual)

        print("Flow:")
        print(graph - graph_residual)

        path_lens = np.array(path_lens)

        plt.subplot(1, 2, 1)
        plt.title("Size of paths found in BFS (in order)")
        plt.plot(path_lens)

        plt.subplot(1, 2, 2)
        plt.title("first-order difference")
        plt.plot(np.diff(path_lens, n=1))
        plt.hlines(
            y=0,
            xmin=0,
            xmax=path_lens.size - 2,
            linestyle="--",
            color="orange")

        plt.show()

    return max_flow


def _experiment_01():
    """Experiment 01."""
    graph = np.array([
        [0, 16, 13, 0, 0, 0],
        [0, 0, 0, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0],
    ])
    max_flow = edkarp_maxflow(graph, 0, graph.shape[0] - 1, verbose=True)
    print("Max flow:", max_flow)


def _experiment_02():
    """Experiment 02."""
    graph = np.array([
        [1, 1e+6, 1e+6, 0],
        [0, 1, 1, 1e+6],
        [0, 0, 0, 1e+6],
        [0, 0, 0, 0],
    ])
    max_flow = edkarp_maxflow(graph, 0, graph.shape[0] - 1, verbose=True)
    print("Max flow:", max_flow)


def _experiment_03():
    """Experiment 03."""
    graph = np.array([
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    max_flow = edkarp_maxflow(
        graph, np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8]), verbose=True)
    print("Max flow:", max_flow)


def _experiment_04():
    """Experiment 04."""
    np.random.seed(16)
    graph = np.random.randint(1, 20, size=(500, 500))
    graph[np.tril_indices(graph.shape[0])] = 0
    graph[np.triu_indices(graph.shape[0], k=4)] = 0
    max_flow = edkarp_maxflow(
        graph,
        np.random.randint(0, 20, size=9),
        graph.shape[0] - 1,
        verbose=True)
    print("Max flow:", max_flow)


def _experiment_06():
    """Experiment 06."""

    def _exclusive_traceback(
            graph: np.ndarray, predecessor_vec: np.ndarray, id_root: int,
            id_target: int) -> t.Tuple[np.ndarray, t.Union[int, float]]:
        """Use the predecessor vector to build the found path."""
        ans = [id_target]
        id_cur_node = id_target
        bottleneck_val = np.inf

        while id_cur_node != id_root:
            prev_node = predecessor_vec[id_cur_node]
            bottleneck_val = min(bottleneck_val, graph[prev_node, id_cur_node])

            # Remove in-out edges from graph to keep then mutually exclusive
            graph[prev_node, id_cur_node] -= 1

            id_cur_node = prev_node
            ans.insert(0, id_cur_node)

        return np.array(ans), bottleneck_val

    def _all_paths_dfs(
            graph: np.ndarray, id_root: int, id_target: int
    ) -> t.Tuple[t.Optional[np.ndarray], t.Union[int, float]]:
        """."""
        queue = [id_root]

        predecessor_vec = np.full(graph.shape[0], -1)
        predecessor_vec[id_root] = -2  # Another invalid value

        while queue:
            id_cur_node = queue.pop()

            if id_cur_node == id_target:
                return _exclusive_traceback(graph, predecessor_vec, id_root,
                                            id_target)

            for id_adj_node, edge_weight in enumerate(graph[id_cur_node]):
                if edge_weight > 0 and predecessor_vec[id_adj_node] == -1:
                    predecessor_vec[id_adj_node] = id_cur_node
                    queue.append(id_adj_node)

        return None, 0

    for i in np.arange(1):
        np.random.seed(16 * (10 + i))

        num_vert = np.random.randint(6, 7)

        graph = np.random.randint(0, 5, size=(num_vert, num_vert))
        graph[np.tril_indices(graph.shape[0])] = 0
        graph[np.triu_indices(graph.shape[0], k=10)] = 0

        graph = 10 * np.array([
            [0, 1, 1, 0, 0, 1],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
        ])
        num_ver = graph.shape[0]

        print(graph)
        graph = extend_vertices_capacity(
            graph,
            np.concatenate(([99 * num_vert], np.ones(num_vert - 2),
                            [99 * num_vert])))
        print(graph)

        id_root = 0
        id_sink = graph.shape[0] - 1

        max_flow = edkarp_maxflow(graph, id_root, id_sink, verbose=False)

        found_paths = 0
        stop = False

        while not stop:
            path, _ = _all_paths_dfs(graph, id_root, id_sink)
            print(path)
            if path is not None:
                found_paths += 1
            else:
                stop = True

        print("Max flow:", max_flow)
        print("Paths:", found_paths)
        assert found_paths == int(max_flow)


def _experiment_05():
    """Experiment 05."""
    graph = np.array([
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ])
    max_flow = edkarp_maxflow(
        graph, np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]), verbose=True)
    print("Max flow:", max_flow)


if __name__ == "__main__":
    # _experiment_01()
    # _experiment_02()
    # _experiment_03()
    # _experiment_04()
    # _experiment_05()
    _experiment_06()
