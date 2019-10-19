"""Finds k vertex/edge disjoint paths using Flow Networks."""
import typing as t

import numpy as np

import ford_fulk_edmo_karp


def _traceback(graph: np.ndarray, predecessor_vec: np.ndarray, id_source: int,
               id_target: int) -> np.ndarray:
    """Use the predecessor vector to build the found path."""
    ans = [id_target]
    id_cur_node = id_target

    while id_cur_node != id_source:
        id_cur_node = predecessor_vec[id_cur_node]
        ans.insert(0, id_cur_node)

    return np.array(ans)


def _bfs(graph: np.ndarray, id_source: int, id_target: int
         ) -> t.Optional[np.ndarray]:
    """Breadth-first search from node ``id_source`` to node ``id_target``."""
    queue = [id_source]

    predecessor_vec = np.full(graph.shape[0], -1)
    predecessor_vec[id_source] = -2  # Another invalid value

    while queue:
        id_cur_node = queue.pop()

        if id_cur_node == id_target:
            return _traceback(graph, predecessor_vec, id_source, id_target)

        for id_adj_node, edge_weight in enumerate(graph[id_cur_node]):
            if edge_weight > 0 and predecessor_vec[id_adj_node] == -1:
                predecessor_vec[id_adj_node] = id_cur_node
                queue.insert(0, id_adj_node)

    return None


def extend_vertices_capacity(graph: np.ndarray,
                             vertex_caps: np.ndarray) -> np.ndarray:
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
    new_graph[connect_related] = vertex_caps

    vals = np.arange(0, 2 * num_vert, 2)
    X, Y = np.meshgrid(vals, vals)

    new_graph[Y + 1, X] = graph

    return new_graph


def find_k_edge_disjoint_paths(
        graph: np.ndarray,
        id_source: int,
        id_sink: int,
        k: int = -1,
        check_edge_weights: bool = True,
) -> t.List[np.ndarray]:
    """Find ``k`` edge-disjoint paths in the given graph if possible.

    Arguments
    ---------
    graph : :obj:`np.ndarray`
        Adjacency matrix of a DAG (Directed Acyclic Graph). The costs
        must be integers.

    id_source : :obj:`int`
        Index of the source in the adjacency matrix.

    id_sink : :obj:`int`
        Same as above, but for the sink(s).

    k : :obj:`int`, optional
        Number of edge-disjoint paths to found. If positive, note that the
        maximum flow of ``graph`` must be at least ``k``. If given a
        negative value, then this algorithm will find the maximum possible
        edge-disjoint paths.

    check_edge_weights : :obj:`bool`, optional
        If True, convert all edge positive weights to 1.

    Notes
    -----
    The paths are found using flow networks.

    Uses Edmond-Karp algorithm to calculate the flow.

    The maximum flow is equal to the edge-disjoint paths in the given graph.

    Uses BFS (Breadth-First Search) to find each path in the flow network.
    """

    def _decrease_flow(flow_graph: np.ndarray, path: t.Sequence[int]) -> None:
        """Decrease flow by one unit to every edge in ``path``."""
        vert_id_prev = path[0]
        for vert_id_cur in path[1:]:
            flow_graph[vert_id_prev, vert_id_cur] -= 1
            vert_id_prev = vert_id_cur

    if check_edge_weights:
        graph[graph > 0] = 1

    max_flow, flow_graph = ford_fulk_edmo_karp.edkarp_maxflow(
        graph=graph,
        id_source=id_source,
        id_sink=id_sink,
        verbose=False,
        return_flow_graph=True)  # type: t.Tuple[int, np.ndarray]

    if k < 0:
        k = max_flow

    if max_flow < k:
        raise ValueError(
            "'There is only {} < k (k = {}) vertex-disjoint paths in this graph."
            .format(max_flow, k))

    paths = []
    for i in np.arange(k):
        path = _bfs(flow_graph, id_source, id_sink)
        _decrease_flow(flow_graph, path)
        paths.append(path)

    return paths


def find_k_vertex_disjoint_paths(
        graph: np.ndarray,
        id_source: int,
        id_sink: int,
        k: int = -1,
) -> t.List[np.ndarray]:
    """Find k vertex disjoint paths in the given graph if possible."""

    def _translate_path(path: t.Sequence[int]) -> np.ndarray:
        """Translate the flow network path to the original graph path."""
        return np.array(path[::2], dtype=int) // 2

    num_vert = graph.shape[0]

    graph = extend_vertices_capacity(
        graph=graph,
        vertex_caps=np.concatenate(([num_vert], np.ones(num_vert - 2),
                                    [num_vert])))

    paths = find_k_edge_disjoint_paths(
        graph=graph,
        k=k,
        id_source=2 * id_source,
        id_sink=2 * id_sink + 1,
        check_edge_weights=False)

    for i in np.arange(len(paths)):
        paths[i] = _translate_path(paths[i])

    return paths



def _test_01():
    """Vertex-disjoint paths."""
    num_tests = 1000
    for i in np.arange(num_tests):
        np.random.seed(8 * (i + 1))

        num_vert = np.random.randint(2, 128)
        graph = np.random.randint(0, 8, size=(num_vert, num_vert))
        graph[np.tril_indices(graph.shape[0])] = 0
        graph[np.triu_indices(graph.shape[0], k=10)] = 0

        paths = find_k_vertex_disjoint_paths(
            graph, id_source=0, id_sink=num_vert - 1)

        used_vertices = set()
        for p in paths:
            _aux = p[1:-1]
            assert used_vertices.isdisjoint(
                _aux) and p[0] == 0 and p[-1] == num_vert - 1
            used_vertices.update(_aux)

        print("\rTest progress: {}%...".format(100 * i / num_tests), end="")

    print("\rRandomized test done.")


def _test_02():
    """Edge-disjoint paths."""
    num_tests = 1000
    for i in np.arange(num_tests):
        np.random.seed(8 * (i + 1))

        num_vert = np.random.randint(2, 128)
        graph = np.random.randint(0, 8, size=(num_vert, num_vert))
        graph[np.tril_indices(graph.shape[0])] = 0
        graph[np.triu_indices(graph.shape[0], k=10)] = 0

        paths = find_k_edge_disjoint_paths(
            graph, id_source=0, id_sink=num_vert - 1)

        used_edges = set()
        for p in paths:
            edges = {(v_i, v_j) for v_i, v_j in zip(p[:-1], p[1:])}
            assert used_edges.isdisjoint(
                edges) and p[0] == 0 and p[-1] == num_vert - 1
            used_edges.update(edges)

        print("\rTest progress: {}%...".format(100 * i / num_tests), end="")

    print("\rRandomized test done.")


if __name__ == "__main__":
    _test_01()
    _test_02()
