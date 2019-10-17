"""Solves the 0-1 Knapsack Problem using Dynamic Programming.

In this variant, a item can either be fully picked or not be picked.
You can pick a fraction of some item.
"""
import typing as t

import numpy as np


def _traceback(
        tabulation: np.ndarray, knapsack_limit: t.Union[int, float],
        item_weights: t.Sequence[t.Union[int, float]],
        item_values: t.Sequence[t.Union[int, float]]) -> t.Sequence[int]:
    """Find the chosen items in the given 0-1 Knapsack Solution."""
    num_items = tabulation.shape[0] - 1
    chosen_items = []  # type: t.List[int]
    cur_value = tabulation[-1, -1]

    cur_item_ind = num_items - 1
    cur_weight = knapsack_limit
    while cur_item_ind >= 0 and cur_value > 0:
        if tabulation[cur_item_ind, cur_weight] == cur_value:
            cur_item_ind -= 1

        else:
            cur_weight -= item_weights[cur_item_ind]
            cur_value -= item_values[cur_item_ind]
            chosen_items.append(cur_item_ind)

    return chosen_items


def zero_one_knapsack(
        knapsack_limit: t.Union[int, float],
        item_weights: t.Sequence[t.Union[int, float]],
        item_values: t.Sequence[t.Union[int, float]],
        return_picked_items: bool = False,
) -> t.Union[float, t.Tuple[float, t.Sequence[int]]]:
    """Solves the 0-1 Knapsack Problem with Dynamic Programming."""
    num_items = len(item_weights)

    if num_items != len(item_values):
        raise ValueError(
            "Length of 'item_weights' ({}) and 'item_values' ({}) "
            "must match.".format(num_items, len(item_values)))

    tabulation = np.zeros((num_items + 1, knapsack_limit + 1), dtype=np.float)

    _knapsack_range = np.arange(1, knapsack_limit + 1)

    for item_ind in np.arange(num_items):
        for k_lim in _knapsack_range:
            tabulation[item_ind + 1, k_lim] = tabulation[item_ind, k_lim]

            if item_weights[item_ind] <= k_lim:
                cur_sum = item_values[item_ind] + tabulation[
                    item_ind, k_lim - item_weights[item_ind]]
                if cur_sum >= tabulation[item_ind, k_lim]:
                    tabulation[item_ind + 1, k_lim] = cur_sum

    best_value = tabulation[num_items, knapsack_limit]

    if return_picked_items:
        chosen_items = _traceback(
            tabulation=tabulation,
            knapsack_limit=knapsack_limit,
            item_weights=item_weights,
            item_values=item_values)

        return best_value, chosen_items

    return best_value


def _test():
    best_value, chosen_items = zero_one_knapsack(
        knapsack_limit=50,
        item_weights=[10, 20, 30],
        item_values=[60, 100, 120],
        return_picked_items=True)

    print("Best value:", best_value)
    print("Chosen items index:", chosen_items)


if __name__ == "__main__":
    _test()
