"""Typical BFS (Breadth-First Search) algorithm."""
import typing as t

import numpy as np


def _traceback(ind_source: int, ind_target: int,
               predecessor_vec: t.Sequence[int]) -> np.ndarray:
    """Build path from source to target using predecessor information."""
    path = [ind_target]
    cur_node = ind_target

    while cur_node != ind_source:
        cur_node = predecessor_vec[cur_node]
        path.insert(0, cur_node)

    return np.array(path)


def _bfs(graph: np.ndarray, ind_source: int, ind_target: t.Optional[int],
         predecessor_vec: t.List[int],
         dist_from_source: t.List[int]) -> t.Optional[np.ndarray]:
    """Perform BFS search."""
    queue = [ind_source]

    while queue:
        cur_node_ind = queue.pop()

        if ind_target is not None and cur_node_ind == ind_target:
            return _traceback(ind_source, ind_target, predecessor_vec)

        for adj_node_ind, weight in enumerate(graph[cur_node_ind, :]):
            if weight > 0 and predecessor_vec[adj_node_ind] == -1:
                # Unlike the DFS, we can set the predecessor right here
                predecessor_vec[adj_node_ind] = cur_node_ind

                dist_from_source[
                    adj_node_ind] = 1 + dist_from_source[cur_node_ind]

                queue.insert(0, adj_node_ind)

    return None


def bfs(graph: np.ndarray,
        ind_source: int,
        ind_target: t.Optional[int] = None,
        return_dists: bool = False,
        random_seed: t.Optional[int] = None
        ) -> t.Union[t.Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Classical Breadth-First Search (BFS) Algorithm.

    Arguments
    ---------
    graph : :obj:`np.ndarray` (N, N)
        Adjacency matrix representing some graph. Only positive
        edges are considered as connectiongs. This matrix must
        be square.

    ind_source : :obj:`int`
        Index of the first source to perform the search. If
        ``ind_target`` is None, then the next source nodes
        are chosen uniformily random among all unreachable
        nodes.

    ind_target : :obj:`int`, optional
        Index of the target node. If not given (None), then
        this BFS will visited all nodes in the given graph,
        chosing randomly (with uniform probability) unreachable
        nodes as the next source nodes.

    return_dists : :obj:`bool`, optional
        If True, return an array with the number of edges between
        each node and some source node.

    random_seed : :obj:`int`, optional
        Set numpy random seed before the first BFS. Useful if
        ``ind_target`` is None.

    Returns
    -------
    If ``return_dists`` is False:
        :obj:`np.ndarray`
            The content of the array depends on the arguments.

        If ``ind_source`` is None:
            Array of predecessor indices, forming a BFS Forest. The
            source nodes predecessor index is ``-1``.

        Else:
            Path from ``ind_source`` and ``ind_target``. If ``graph``
            represents a unweighted graph, then this is one shortest-
            path between these two nodes.

    Else:
        :obj:`tuple` (:obj:`np.ndarray`, :obj:`np.ndarray`)
            The first numpy array is the same as if ``return_dists`` is
            False. The second numpy array have the number of eadges
            between the corresponding node and some source node.
    """
    num_nodes = graph.shape[0]

    if num_nodes != graph.shape[1]:
        raise ValueError("Graph adjacency matrix must be square. (Got "
                         "shape {}.)".format(graph.shape))

    if not 0 <= ind_source < num_nodes:
        raise ValueError("'ind_source' must be in [0, num_nodes) (got {}.)".
                         format(ind_source))

    if ind_target is not None and not 0 <= ind_target < num_nodes:
        raise ValueError("'ind_target' must be None or be in [0, num_nodes) "
                         "(got {}.)".format(ind_target))

    if random_seed is not None:
        np.random.seed(random_seed)

    predecessor_vec = np.full(num_nodes, -1, dtype=int)
    dist_from_source = np.zeros(num_nodes, dtype=int)

    _single_it = ind_target is not None
    _discover_all_criteria = ind_target is None

    while _discover_all_criteria or _single_it:
        _single_it = False

        predecessor_vec[
            ind_source] = -2  # Note: to differ source from undiscovered nodes
        path = _bfs(
            graph=graph,
            ind_source=ind_source,
            ind_target=ind_target,
            predecessor_vec=predecessor_vec,
            dist_from_source=dist_from_source)

        _discover_all_criteria = ind_target is None and -1 in predecessor_vec

        if _discover_all_criteria:
            ind_source = np.random.choice(np.where(predecessor_vec == -1)[0])

    if ind_target is None:
        predecessor_vec[predecessor_vec < 0] = -1
        ret = predecessor_vec

    else:
        ret = path

    if return_dists:
        ret = (ret, dist_from_source)

    return ret


def _test_01():
    graph = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 0],
    ])

    path = bfs(graph=graph, ind_source=0, ind_target=graph.shape[0] - 1)

    print("Path:", path)


def _test_02():
    graph = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])

    path, dists = bfs(
        graph=graph,
        ind_source=0,
        ind_target=None,
        random_seed=16,
        return_dists=True)

    print("Path:", path)
    print(dists)


if __name__ == "__main__":
    _test_01()
    _test_02()
