"""Approximation for NP-Complete Vertex Cover problem."""
import typing as t


def approx_vertex_cover(
        graph: t.Dict[t.Union[int, str], t.Iterable[t.Union[int, str]]]
) -> t.Set[t.Union[int, str]]:
    r"""Find a approximated vertex cover in undirected ``graph``.

    A Vertex Cover of a graph G = (V, E) is a subset $V' \in V$
    sush that for every $(u, v) \in E$, $u \in V'$ or $v \in V'$
    or both.

    This is a optimization problem, and the goal is to find the
    smallest vertex cover possible in G. This is a NP-Complete
    problem and, therefore, is unknown is there exists a polynomial
    time algorithm that can solve this problem efficiently.

    However, this algorithm is an approximation algorithm and
    thus can provide an suboptimal solution that is guaranteed
    to be at most twice worse than the optimal solution (i.e.
    the solution set may have twice as much vertices than the
    optimal solution in the worst-case scenario.)

    Technically speaking, this is a 2-approximation algorithm.
    """

    def _check_self_loop(vert_a, vert_b):
        return vert_a != vert_b

    vert_cov = set()  # type: t.Set[t.Union[int, str]]

    graph_copy = {vertex: set(graph[vertex]) for vertex in graph}

    while graph_copy:
        rand_vertex_a, adj_vertex = graph_copy.popitem()

        remaining_edges = adj_vertex - vert_cov

        if remaining_edges:
            rand_vertex_b = remaining_edges.pop()

            if _check_self_loop(rand_vertex_a, rand_vertex_b):
                graph_copy.pop(rand_vertex_b)

            vert_cov.update({rand_vertex_a, rand_vertex_b})

    return vert_cov


def _test_01():
    """Test with a example graph from Cormen et al."""
    graph = {
        'a': ['b'],
        'b': ['a', 'c'],
        'c': ['b', 'e', 'd'],
        'd': ['e', 'f', 'g'],
        'e': ['c', 'd', 'f'],
        'f': ['e', 'd'],
        'g': ['d'],
    }

    ver_cov = approx_vertex_cover(graph=graph)

    print(
        "(Possibly) suboptimal vertex cover (size: {}):".format(len(ver_cov)),
        ver_cov)


def _test_02():
    """Test with a circular graph."""
    graph_size = 100
    graph = {
        i: [i + 1]
        for i in range(1, graph_size)
    }
    graph[graph_size] = [1]

    ver_cov = approx_vertex_cover(graph=graph)

    print(
        "(Possibly) suboptimal vertex cover (size: {}):".format(len(ver_cov)),
        ver_cov)


if __name__ == "__main__":
    # _test_01()
    _test_02()
