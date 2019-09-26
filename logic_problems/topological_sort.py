import warnings
import enum


class NodeState(enum.IntEnum):
    NOT_PROCESSED = 1
    IN_PROGRESS = 2
    PROCESSED = 3


def topological_sort3(graph):
    """Topological sort using DFS."""

    def _dfs(node, graph, node_state, ans):
        if node_state[node] == NodeState.IN_PROGRESS:
            return False

        if node_state[node] == NodeState.NOT_PROCESSED:
            node_state[node] = NodeState.IN_PROGRESS
            for adj_node in graph[node]:
                if not _dfs(adj_node, graph, node_state, ans):
                    return False

            node_state[node] = NodeState.PROCESSED
            ans.insert(0, node)

        return True

    ans = []

    node_state = {node: NodeState.NOT_PROCESSED for node in graph}
    for node in graph:
        if node_state[node] == NodeState.NOT_PROCESSED:
            if not _dfs(node, graph, node_state, ans):
                warnings.warn(
                    "Cycle detected in graph. "
                    "Can't do topological sort", UserWarning)

                return None

    return ans


def topological_sort2(graph):
    """Topological sort using dependency count V2."""
    ALREADY_PROCESSED = -1

    ans = []
    dep_count = {node: 0 for node in graph}
    for adj_list in graph.values():
        for adj_node in adj_list:
            dep_count[adj_node] += 1

    break_flag = True
    while break_flag:
        break_flag = False
        for node in dep_count:
            if dep_count[node] == 0:
                dep_count[node] = ALREADY_PROCESSED
                ans.append(node)
                for adj_node in graph[node]:
                    dep_count[adj_node] -= 1

                break_flag = True

    if len(ans) == len(graph):
        return ans

    warnings.warn("Cycle detected in graph. "
                  "Can't do topological sort", UserWarning)

    return None


def topological_sort1(graph):
    """Topological sort using dependency count V1."""
    ALREADY_PROCESSED = -1

    inc_level = {node: 0 for node in graph}
    topological_list = []

    for adj_nodes in graph.values():
        for node in adj_nodes:
            inc_level[node] += 1

    process_queue = []
    for node in graph:
        if inc_level[node] == 0:
            process_queue.insert(0, node)
            inc_level[node] = -1

    while process_queue:
        cur_node = process_queue.pop()

        for adj_node in graph[cur_node]:
            inc_level[adj_node] -= 1

        topological_list.append(cur_node)

        for node in inc_level:
            if inc_level[node] == 0:
                process_queue.insert(0, node)
                inc_level[node] = ALREADY_PROCESSED

    if len(topological_list) == len(graph):
        return topological_list

    warnings.warn("Cycle detected in graph. "
                  "Can't do topological sort", UserWarning)

    return None


if __name__ == "__main__":
    graph = {
        "a": ["b", "c"],
        "b": ["d"],
        "c": ["d"],
        "d": ["e"],
        "e": [],
    }

    print(topological_sort1(graph))
    print(topological_sort2(graph))

    graph = {
        "a": ["b", "c"],
        "b": ["d"],
        "c": ["d"],
        "d": ["e"],
        "e": ["b"],
    }

    print(topological_sort1(graph))
    print(topological_sort2(graph))

    graph = {
        0: [],
        1: [],
        2: [3],
        3: [1],
        4: [0, 1],
        5: [2, 0],
    }

    print(topological_sort1(graph))
    print(topological_sort2(graph))
