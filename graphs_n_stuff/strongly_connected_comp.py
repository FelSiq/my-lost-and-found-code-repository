"""."""
import typing as t
import enum

import numpy as np

_time = 0
"""Variable to keep track of the DFS timestamps."""


class TStampIndex(enum.IntEnum):
    """Indices for the timestamp table of DFS."""
    DISCOVERY = 0
    EXPLORED = 1


def _dfs(graph: np.ndarray, ind_cur_node: int, timestamps: np.ndarray,
         colors: t.List[int], color: int) -> None:
    global _time

    timestamps[TStampIndex.DISCOVERY, ind_cur_node] = _time
    _time += 1

    colors[ind_cur_node] = color

    for ind_adj_node, weight in enumerate(graph[ind_cur_node, :]):
        if weight > 0 and timestamps[TStampIndex.
                                     DISCOVERY, ind_adj_node] == -1:
            _dfs(graph, ind_adj_node, timestamps, colors, color)

    timestamps[TStampIndex.EXPLORED, ind_cur_node] = _time
    _time += 1


def dfs(graph: np.ndarray, node_seq: t.Sequence[int],
        return_colors: bool) -> np.ndarray:
    """Typical Depth-First Search (DFS) which returns final timestamps.

    The final timestamps are the number of steps necessary to finish
    exploring the corresponding node.
    """
    num_nodes = graph.shape[0]

    if num_nodes != graph.shape[1]:
        raise ValueError("Graph adjacency matrix must be square (got shape "
                         "{}.)".format(graph.shape))

    timestamps = np.full((2, num_nodes), -1, dtype=int)
    colors = np.full(num_nodes, -1, dtype=int)

    global _time
    _time = 0

    color = 0
    for source_node_ind in node_seq:
        if timestamps[TStampIndex.DISCOVERY, source_node_ind] == -1:
            _dfs(graph, source_node_ind, timestamps, colors, color)
            color += 1

    if return_colors:
        return colors

    return timestamps[TStampIndex.EXPLORED, :]


def strongcc(graph: np.ndarray) -> t.Tuple[int, ...]:
    """Compute the strongly connected components.
    
    Arguments
    ---------
    graph : :obj:`np.ndarray`
        Adjacency matrix representing a graph.

    Returns
    -------
    :obj:`tuple` of :obj:`int`
        Tuple addressing a integral value to every vertex, corresponding
        to the index of its strongly connected component cluster.
    """
    num_nodes = graph.shape[0]

    final_timestamps = dfs(
        graph=graph, node_seq=np.arange(num_nodes), return_colors=False)

    scc_index = dfs(
        graph=graph.T,
        node_seq=np.argsort(-final_timestamps),
        return_colors=True)

    return tuple(scc_index)


def _test() -> None:
    graph = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ])
    scc = strongcc(graph=graph)
    print(scc)


if __name__ == "__main__":
    _test()
