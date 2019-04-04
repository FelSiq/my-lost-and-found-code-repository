import copy


def stable_marriage(group_1, group_2):
    """Generates a stable matching, but it is unfair as it benefits group_1.

    Def.: (Matching) A matching in a graph G is a subset of edges of G such
    that no two edges share incidence in some same vertex (in other words,
    the graph is shattered into components that has at most two vertices
    in it).

    Def.: (Rogue Couple) Two vertices, V and U, are said to form a Rogue
    Couple in a matching if they both prefer each other better than the
    current their matched pairs.

    Def.: (Stable matching) A Matching is said to be Stable if there's no
    Rogue Couples within.

    Def.: (Realm of Possible Choices) One vertex V is said to be in the
    Realm of Possible Choices in U if there exists a stable matching such
    that U is matched with V.

    More precisely, all elements of group_1 are always matched with its
    optimal pair in its Realm of Possible Choices, and all elements in
    group_2 are always matched with the pessimal pair in its Realm of
    Possible Choices.

    Def.: (Perfect matching) A Matching is said to be Perfect if every
    node of G is incident to some edge in the Matching (i.e., all nodes
    of the graph are matched.)

    This algorithm always produces a Perfect matching given that the
    two groups have the same cardinality (i.e., the same number of ele-
    ments.)
    """
    match = {v_2: (None, None) for v_2 in group_2}
    preferences = copy.deepcopy(group_1)

    while (None, None) in match.values():
        for v_1 in group_1:
            candidate = preferences[v_1][0]
            v_1_pref = group_2[candidate].index(v_1)
            rival, rival_pref = match[candidate]

            # Note: the "less" the preference, the better (preference = index)
            if rival is None or (v_1 != rival and rival_pref > v_1_pref):
                match[candidate] = (v_1, v_1_pref)
                if rival:
                    preferences[rival].remove(candidate)

    return match


if __name__ == "__main__":
    g_1 = {
        "Brad": ["Jennifer", "Angelina"],
        "Billy Bob": ["Jennifer", "Angelina"],
    }

    g_2 = {
        "Jennifer": ["Billy Bob", "Brad"],
        "Angelina": ["Billy Bob", "Brad"],
    }

    ans = stable_marriage(g_1, g_2)
    print(ans)

    ans = stable_marriage(g_2, g_1)
    print(ans)
