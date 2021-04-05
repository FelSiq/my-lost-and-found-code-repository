"""Tarjan algorithm to find articulation points given a graph.

TarjanNodes find articulation points (nodes).

TarjanEdges find critical connections (edges).
"""


class TarjanNodes:
    def __init__(self, graph: t.Dict[t.Any, t.List[t.Any]]):
        self.graph = graph

        self.time_first_visit = None
        self.time_min = None
        self.artic_points = None
        self.time = -1

    def _dfs(self, node: t.Any, parent: t.Any):
        self.time += 1
        self.time_first_visit[node] = self.time_min[node] = self.time

        indep_children = 0

        for adj_node in self.graph[node]:
            if adj_node == parent:
                continue

            if self.time_first_visit.get(adj_node, 0) == 0:
                indep_children += 1
                self._dfs(adj_node, node)

                if self.time_first_visit[node] <= self.time_min[adj_node]:
                    self.artic_points.append(node)

                self.time_min[node] = min(self.time_min[node], self.time_min[adj_node])

            else:
                self.time_min[node] = min(
                    self.time_min[node], self.time_first_visit[adj_node]
                )

        return indep_children

    def run(self):
        self.colors = dict()
        self.time_first_visit = dict()
        self.time_min = dict()

        self.artic_points = []

        self.time = 0

        for v in self.graph:
            if v not in self.time_first_visit:
                if self._dfs(v, None) > 1:
                    self.artic_points.append(v)

        return self.artic_points


class TarjanEdges:
    def __init__(self, graph: t.Dict[t.Any, t.List[t.Any]]):
        self.graph = graph

        self.time_first_visit = None
        self.time_min = None
        self.artic_points = None
        self.time = -1

    def _dfs(self, node: t.Any, parent: t.Any):
        self.time += 1
        self.time_first_visit[node] = self.time_min[node] = self.time

        for adj_node in self.graph[node]:
            if adj_node == parent:
                continue

            if self.time_first_visit.get(adj_node, 0) == 0:
                self._dfs(adj_node, node)

                if self.time_first_visit[node] < self.time_min[adj_node]:
                    self.artic_points.append([node, adj_node])

                self.time_min[node] = min(self.time_min[node], self.time_min[adj_node])

            else:
                self.time_min[node] = min(
                    self.time_min[node], self.time_first_visit[adj_node]
                )

    def run(self):
        self.colors = dict()
        self.time_first_visit = dict()
        self.time_min = dict()

        self.artic_points = []

        self.time = 0

        for v in self.graph:
            if v not in self.time_first_visit:
                self._dfs(v, None)

        return self.artic_points
