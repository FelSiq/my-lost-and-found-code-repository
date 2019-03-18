import warnings


def topological_sort(graph):
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
                inc_level[node] = -1

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

    print(topological_sort(graph))

    graph = {
        "a": ["b", "c"],
        "b": ["d"],
        "c": ["d"],
        "d": ["e"],
        "e": ["b"],
    }

    print(topological_sort(graph))

    graph = {
        0: [],
        1: [],
        2: [3],
        3: [1],
        4: [0, 1],
        5: [2, 0],
    }

    print(topological_sort(graph))
