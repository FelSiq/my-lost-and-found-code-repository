"""Naive implementation of an approximation algorithm for Set Cover problem.

Set Cover problem is NP-Complete and, thus, it is unlikely that a
efficient (polynomial time) solution exists to solve it exactly.
"""
import typing as t

SetElementType = t.Union[str, int]


def greedy_set_cover(set_family: t.Sequence[t.Set[SetElementType]],
                     all_elements: t.Optional[t.Set[SetElementType]] = None
                     ) -> t.List[t.Set[SetElementType]]:
    """Naive implementation of set cover. Just for reference.

    There exists a linear time solution.
    """
    if all_elements is None:
        all_elements = set()
        all_elements.update(*set_family)

    else:
        all_elements = all_elements.copy()

    set_cover = []
    not_merged_inds = [i for i in range(len(set_family))]

    while all_elements:
        max_diff_merge_ind, max_diff_merge_val = 0, 0

        for ind in not_merged_inds:
            diff_merge_val = len(all_elements.intersection(set_family[ind]))

            if diff_merge_val > max_diff_merge_val:
                max_diff_merge_val = diff_merge_val
                max_diff_merge_ind = ind

        print(max_diff_merge_ind)
        chosen_set = set_family[max_diff_merge_ind]
        all_elements -= chosen_set
        set_cover.append(chosen_set)
        not_merged_inds.remove(max_diff_merge_ind)

    return set_cover


def _test():
    """Example from Cormen et. al 3rd Edition (page 1118)."""
    set_family = [
        {1, 2, 3, 4, 5, 6},
        {5, 6, 8, 9},
        {1, 4, 7, 10},
        {2, 5, 8, 11, 7},
        {3, 6, 9, 12},
        {10, 11},
    ]

    print("Greedy set cover:", greedy_set_cover(set_family))


if __name__ == "__main__":
    _test()
