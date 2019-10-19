"""Given a DAG, sample uniformly a path from two given vertices.

DAG stands for ``Directed Acyclic Graph``.

Because the sampling follows a uniform distribution, all paths
between two given vertices must have the same probability of
being chosen.
"""
import typing as t
import heapq

import numpy as np


class DAGPathSampler:
    def __init__(self):
        self.node_in_num_path = None  # type: t.Optional[np.ndarray]
        self.graph = None  # type: t.Optional[np.ndarray]
        self.num_nodes = -1
        self.ind_source = -1
        self.ind_target = -1

    def _topological_sort(self, graph: np.ndarray) -> np.ndarray:
        in_degree = graph.sum(axis=0)
        queue = [self.ind_source]

        top_sort = np.full(self.num_nodes, -1, dtype=int)
        cur_ind = 0
        while queue:
            cur_node_ind = queue.pop()

            in_degree[graph[cur_node_ind, :] > 0] -= 1
            in_degree[cur_node_ind] = -1
            top_sort[cur_ind] = cur_node_ind
            cur_ind += 1

            for fulfilled_node_ind in np.where(in_degree == 0)[0]:
                queue.insert(0, fulfilled_node_ind)
                in_degree[fulfilled_node_ind] = -1

        if cur_ind != self.num_nodes:
            raise ValueError(
                "Can't perform topological sort, graph is not a DAG.")

        return top_sort

    def _get_num_paths(self, graph: np.ndarray) -> np.ndarray:
        top_sort = self._topological_sort(graph)

        num_paths = np.zeros(self.num_nodes)
        num_paths[self.ind_source] = 1

        for node_id in top_sort:
            num_paths[node_id] += np.sum(num_paths[graph[:, node_id] > 0])

        return num_paths

    def calculate_probs(self, graph: np.ndarray, ind_source: int,
                        ind_target: int) -> "DAGPathSampler":
        """Calculate the probability of each edge be chosen.

        Arguments
        ---------

        Returns
        -------
        """
        self.num_nodes = graph.shape[0]

        if self.num_nodes != graph.shape[1]:
            raise ValueError("Graph adjacency matrix must be square! "
                             "(got shape {}.)".format(graph.shape))

        if not 0 <= ind_source < self.num_nodes:
            raise ValueError("Invalid source node (index {}.) Must be "
                             "in range [0, {}].".format(
                                 ind_source, self.num_nodes - 1))

        self.graph = graph > 0
        self.ind_source = ind_source
        self.ind_target = ind_target

        self.node_in_num_path = self._get_num_paths(graph=self.graph)

        return self

    def sample(self, num_paths: int = 1, random_seed: t.Optional[int] = None
               ) -> t.Union[t.Tuple[int, ...], t.Sequence[t.Tuple[int, ...]]]:
        """Sample uniformly a path from the fitted DAG.

        Notes
        -----
        Uses randomized traceback (path is built from the end to start).
        """
        if random_seed is None:
            np.random.seed(random_seed)

        paths = []

        for _ in np.arange(num_paths):
            cur_path = [self.ind_target]
            cur_node = self.ind_target

            while cur_node != self.ind_source:
                adj_in_nodes = np.where(self.graph[:, cur_node] > 0)[0]
                probs = self.node_in_num_path[
                    adj_in_nodes] / self.node_in_num_path[cur_node]
                cur_node = np.random.choice(adj_in_nodes, p=probs)
                cur_path.insert(0, cur_node)

            paths.append(tuple(cur_path))

        if num_paths == 1:
            return paths[0]

        return paths


def _test_01():
    """."""
    import matplotlib.pyplot as plt

    graph = np.array([
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ])
    sampler = DAGPathSampler().calculate_probs(
        graph=graph, ind_source=0, ind_target=graph.shape[0] - 1)

    num_paths = 10000
    paths = sampler.sample(num_paths=num_paths, random_seed=16)
    path_freq = {}  # t.Dict[t.Tuple[int, ...], int]
    for path in paths:
        if path not in path_freq:
            path_freq[path] = 0
        path_freq[path] += 1

    num_distinct_paths = len(path_freq)
    path_ids = np.arange(num_distinct_paths)
    plt.bar(path_ids,
            np.array(list(path_freq.values()), dtype=float) / num_paths)
    plt.hlines(
        y=1 / num_distinct_paths,
        xmin=-1,
        xmax=path_ids.size,
        linestyle="--",
        color="black",
        label="expected")
    plt.legend()
    plt.show()


def _test_02():
    import matplotlib.pyplot as plt

    graph = np.random.randint(0, 5, size=(10, 10))
    graph = np.tril(np.triu(graph, 1), 5)
    sampler = DAGPathSampler().calculate_probs(
        graph=graph, ind_source=0, ind_target=graph.shape[0] - 1)

    num_paths = 20000
    paths = sampler.sample(num_paths=num_paths, random_seed=16)
    path_freq = {}  # t.Dict[t.Tuple[int, ...], int]
    for path in paths:
        if path not in path_freq:
            path_freq[path] = 0
        path_freq[path] += 1

    num_distinct_paths = len(path_freq)
    path_ids = np.arange(num_distinct_paths)
    plt.bar(path_ids,
            np.array(list(path_freq.values()), dtype=float) / num_paths)
    plt.hlines(
        y=1 / num_distinct_paths,
        xmin=-1,
        xmax=path_ids.size,
        linestyle="--",
        color="black",
        label="expected")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _test_01()
    _test_02()
