"""Simple implementation of (almost) equivalent iterative and recursive DFS.

The choice of the implementation method (iterative or recursive) depends
on your application needs. If you need information AFTER some node is
processed, then you probably should consider the recursive form for simplicity.
"""
import typing as t

import numpy as np


_time_it = 0
_time_rec = 0


def _dfs_it(graph: np.ndarray, ind_cur_node: int, predecessor: np.ndarray, timestamps: np.ndarray) -> None:
    """Iterative version of the Depth-First Search (DFS) algorithm."""
    global _time_it

    num_node = graph.shape[0]
    stack = [(ind_cur_node, -1)]

    while stack:
        ind_cur_node, ind_predecessor = stack.pop()

        # Unlike BFS, we need to re-check if the current node is still
        # not visited, as we don't know what happened after it
        # was stacked by some adjacent node (ancestor candidate.)
        if timestamps[0, ind_cur_node] != -1:
            continue

        predecessor[ind_cur_node] = ind_predecessor
        timestamps[0, ind_cur_node] = _time_it

        # There's no way to define the final timestamp of a finished node
        # without an additional data structure. If that information is relevant
        # to the application, then the recursive approach would fit much more
        # naturally.
        _time_it += 1

        for rev_ind_adj_node, weight in enumerate(graph[ind_cur_node, :][::-1], 1):
            # We must iterate the list backwards in order to produce the
            # same results of a recursive approach, because we need to fill
            # the adjacent nodes candidates backwards in the stack in order
            # to the first natural candidate be in the top of the stack.
            ind_adj_node = num_node - rev_ind_adj_node

            if weight > 0 and timestamps[0, ind_adj_node] == -1:
                # Unlike BFS, we don't set the predecessor here, as we don't
                # know yet if the current node will be actually the predecessor
                # of this adjacent node.
                stack.append((ind_adj_node, ind_cur_node))


def _dfs_rec(graph: np.ndarray, ind_cur_node: int, predecessor: np.ndarray, timestamps: np.ndarray) -> None:
    global _time_rec
    timestamps[0, ind_cur_node] = _time_rec
    _time_rec += 1

    for ind_adj_node, weight in enumerate(graph[ind_cur_node, :]):
        if weight > 0 and timestamps[0, ind_adj_node] == -1:
            predecessor[ind_adj_node] = ind_cur_node
            _dfs_rec(graph, ind_adj_node, predecessor, timestamps)
    
    timestamps[1, ind_cur_node] = _time_rec
    _time_rec += 1


def dfs(graph: np.ndarray, ind_source: int, random_seed: t.Optional[int] = None, iterative: bool = True) -> np.ndarray:
    num_nodes = graph.shape[0]
    timestamps = np.full((2 - iterative, num_nodes), -1)
    predecessor = np.full(num_nodes, -1)

    if random_seed is not None:
        np.random.seed(random_seed)

    if iterative:
        global _time_it
        _time_it = 0
        func = _dfs_it

    else:
        global _time_rec
        _time_rec = 0
        func = _dfs_rec

    remaining_nodes = True
    random_sources = np.random.permutation(np.arange(num_nodes))
    random_source_ind = 0

    while remaining_nodes:
        func(graph, ind_source, predecessor, timestamps)

        while random_source_ind < num_nodes and timestamps[0, ind_source] != -1:
            random_source_ind += 1

        remaining_nodes = False
        if random_source_ind < num_nodes:
            ind_source = random_sources[random_source_ind]
            remaining_nodes = True

    return timestamps, predecessor


def _test_01():
    graph = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=int)

    random_seed = 16
    ind_source = 0

    timestamps_it, pred_it = dfs(graph, ind_source, random_seed=random_seed, iterative=True)
    timestamps_rec, pred_rec = dfs(graph, ind_source, random_seed=random_seed, iterative=False)

    assert np.allclose(np.argsort(timestamps_it[0, :]), np.argsort(timestamps_rec[0, :]))
    assert np.allclose(pred_it, pred_rec)

    graph += graph.T

    timestamps_it, pred_it = dfs(graph, ind_source, random_seed=random_seed, iterative=True)
    timestamps_rec, pred_rec = dfs(graph, ind_source, random_seed=random_seed, iterative=False)

    assert np.allclose(np.argsort(timestamps_it[0, :]), np.argsort(timestamps_rec[0, :]))
    assert np.allclose(pred_it, pred_rec)


def _test_02():
    num_it = 20000
    for i in np.arange(num_it):
        random_seed = (1 + i) * 2
        np.random.seed(random_seed)
        graph_dim = np.random.randint(1, 100)
        graph = np.random.randint(3, size=(graph_dim, graph_dim))

        if np.random.randint(2) == 0:
            graph = np.tril(np.triu(graph), k=25)

        ind_source = np.random.randint(graph_dim)

        timestamps_it, pred_it = dfs(graph, ind_source, random_seed=random_seed, iterative=True)
        timestamps_rec, pred_rec = dfs(graph, ind_source, random_seed=random_seed, iterative=False)

        assert np.allclose(np.argsort(timestamps_it[0, :]), np.argsort(timestamps_rec[0, :]))
        assert np.allclose(pred_it, pred_rec)
        print("\rTest: {:.2f}".format(100 * i / num_it), end="")

    print("\rRandomized test done.")


if __name__ == "__main__":
    _test_01()
    _test_02()
