"""Classical recursive Depth-First Search (DFS)."""
import typing as t
import enum

import numpy as np

_time = 0
"""Variable used to fill timestamps in DFS."""


class TStampIndex(enum.IntEnum):
    """Index for Timestamps information about node discovery and finish."""
    DISCOVERY = 0
    FINISHED = 1


def _traceback(predecessor_vec: t.Sequence[int],
               ind_target: int) -> t.List[int]:
    """Build the found path between ``ind_source`` and ``ind_target``."""
    path = [ind_target]
    cur_node = ind_target

    while cur_node >= 0:
        cur_node = predecessor_vec[cur_node]
        path.insert(0, cur_node)

    return path[1:]


def _is_descendant(ind_descendant: int, ind_ancestor: int,
                   timestamps: np.ndarray) -> bool:
    """Checks is ``ind_descendant`` node is descendant of ``ind_ancestor`` node."""
    t_stamp_disc_anc, t_stamp_disc_desc = timestamps[
        TStampIndex.DISCOVERY, [ind_ancestor, ind_descendant]]
    t_stamp_fin_anc, t_stamp_fin_desc = timestamps[
        TStampIndex.FINISHED, [ind_ancestor, ind_descendant]]

    if t_stamp_fin_anc == -1:
        t_stamp_fin_anc = np.inf

    return t_stamp_disc_anc < t_stamp_disc_desc < t_stamp_fin_desc < t_stamp_fin_anc


def _count_edge(ind_source: int, ind_adj_node: int, timestamps: np.ndarray,
                edge_type_counts: t.Dict[str, int]) -> None:
    """Count the edge based on its classification."""
    if timestamps[TStampIndex.DISCOVERY, ind_adj_node] == -1:
        # 'TREE' edges are the adges connecting directly a node and its predecessor
        edge_type_counts["TREE"] += 1

    elif timestamps[TStampIndex.FINISHED, ind_adj_node] == -1:
        # Adj node is an ancestor of the current node in the DFS tree
        # 'BACK' edges connects a node to some of its ancestor, except
        # for its predecessor
        edge_type_counts["BACK"] += 1

    elif timestamps[TStampIndex.DISCOVERY, ind_adj_node] > timestamps[
            TStampIndex.DISCOVERY, ind_source]:
        # Adj node is a descendant of the current node in the DFS tree
        # 'FORWARD' edges connects a node to some of its descendants in
        # the DFS tree, except for its directly descendants connected
        # by a 'TREE' edge
        edge_type_counts["FORWARD"] += 1

    else:
        # Cross edges connects nodes from different subtrees (i.e., neither
        # are descendant nor ancestor of another.)
        edge_type_counts["CROSS"] += 1


def _dfs(graph: np.ndarray, ind_source: int, ind_target: t.Optional[int],
         predecessor_vec: t.List[int], timestamps: np.ndarray,
         edge_type_counts: t.Dict[str, int],
         parenthesis: t.List[str]) -> t.Optional[t.List[int]]:
    """Recursive procedure of DFS.

    Typically, if just the path or the predecessor is useful for
    the application, then it is simpler to just implement the
    iterative version, which is EXACTLY like the BFS algorithm,
    but changing the queue structure for a stack.

    However, in this case, where a lot of extra information (like
    the discovery/finish timestamps and node parenthesization) can
    be useful, then the recursive approach much more convenient.
    """
    if ind_target is not None and ind_source == ind_target:
        return _traceback(
            predecessor_vec=predecessor_vec, ind_target=ind_target)

    global _time

    timestamps[TStampIndex.DISCOVERY, ind_source] = _time
    parenthesis.append("({}".format(ind_source))
    _time += 1

    path = None
    for ind_adj_node, weight in enumerate(graph[ind_source, :]):
        if weight > 0:
            _count_edge(
                edge_type_counts=edge_type_counts,
                timestamps=timestamps,
                ind_source=ind_source,
                ind_adj_node=ind_adj_node)

            if predecessor_vec[ind_adj_node] == -1:
                predecessor_vec[ind_adj_node] = ind_source

                path = _dfs(
                    graph=graph,
                    ind_source=ind_adj_node,
                    ind_target=ind_target,
                    predecessor_vec=predecessor_vec,
                    timestamps=timestamps,
                    edge_type_counts=edge_type_counts,
                    parenthesis=parenthesis)

                if path is not None:
                    break

    timestamps[TStampIndex.FINISHED, ind_source] = _time
    parenthesis.append("{})".format(ind_source))
    _time += 1

    return path


def dfs(graph: np.ndarray,
        ind_source: int,
        ind_target: t.Optional[int] = None,
        return_parenthesis: bool = False,
        return_timestamps: bool = False,
        return_edge_counts: bool = False,
        random_seed: t.Optional[int] = None
        ) -> t.Union[t.Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Classical recursive Depth-First Search (DFS) algorithm."""
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

    global _time
    _time = 0

    predecessor_vec = np.full(num_nodes, -1, dtype=int)
    timestamps = np.full((2, num_nodes), -1, dtype=int)
    edge_type_counts = {
        type_: 0
        for type_ in ["TREE", "BACK", "FORWARD", "CROSS"]
    }
    parenthesis = []  # type: t.List[str]

    _single_it = ind_target is not None
    _discover_all_criteria = ind_target is None

    while _discover_all_criteria or _single_it:
        _single_it = False

        predecessor_vec[
            ind_source] = -2  # Note: to differ source from undiscovered nodes

        path = _dfs(
            graph=graph,
            ind_source=ind_source,
            ind_target=ind_target,
            predecessor_vec=predecessor_vec,
            timestamps=timestamps,
            edge_type_counts=edge_type_counts,
            parenthesis=parenthesis)

        _discover_all_criteria = ind_target is None and -1 in predecessor_vec

        if _discover_all_criteria:
            ind_source = np.random.choice(np.where(predecessor_vec == -1)[0])

    ret = []

    if ind_target is None:
        predecessor_vec[predecessor_vec < 0] = -1
        ret.append(predecessor_vec)

    else:
        ret.append(path)

    if return_parenthesis:
        ret.append("".join(parenthesis))

    if return_timestamps:
        ret.append(timestamps)

    if return_edge_counts:
        ret.append(edge_type_counts)

    return tuple(ret)


def _test():
    graph = np.array([
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0],
    ],
                     dtype=int)

    pred_vec, parenthesis, timestamps, edge_type_counts = dfs(
        graph=graph,
        ind_source=0,
        ind_target=None,
        return_timestamps=True,
        return_edge_counts=True,
        return_parenthesis=True,
        random_seed=None)

    print("DFS forest predecessor vector:", pred_vec)
    print("Parenthesis:", parenthesis)

    print("Timestamps:")
    print("\tStart:", timestamps[TStampIndex.DISCOVERY, :])
    print("\tEnd:", timestamps[TStampIndex.FINISHED, :])

    print("Edge types")
    for edge_type in edge_type_counts:
        print("\t{:<{fill}}: {}".format(
            edge_type, edge_type_counts[edge_type], fill=7))


if __name__ == "__main__":
    _test()
