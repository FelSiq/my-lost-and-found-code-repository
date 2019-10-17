"""Fractional Knapsack Problem (FKP) solved by a Greedy Strategy.

In the FKP variante of the Knapsack Problem, we can take
fractional parts of every item (think the items as ``fluids``
instead of ``solid-state`` objects.)
"""
import typing as t

import numpy as np


def fractional_knapsack(
        knapsack_limit: t.Union[int, float],
        item_weights: t.Sequence[t.Union[int, float]],
        item_values: t.Sequence[t.Union[int, float]],
        item_quantity: t.Optional[t.Sequence[t.Union[int, float]]] = None,
        return_item_quant: bool = False,
) -> t.Union[float, t.Tuple[float, t.Sequence[float]]]:
    """Solves the Fractional Knapsack problem.

    Arguments
    ---------
    knapsack_limit : :obj:`int` or :obj:`float`
        Limit of weight to carry.

    item_weights : :obj:`Sequence` of :obj:`int` or :obj:`float`
        Weight of each corresponding item. All values must be positive
        values.

    item_values : :obj:`Sequence` of :obj:`int' or :obj:`float`
        Values associated with a whole unit of the corresponding item.

    item_quantity: :obj:`Sequence` of :obj:`int` or :obj:`float`, optional
        Sequence of integers representing the quantity of each item.
        If not given (:obj:`NoneType`), then each item quantity is
        assumed to be 1.0.

    return_item_quant : :ob:`bool`, optional
        If True, return the total fraction obtained for each item.

    Returns
    -------
    if ``return_item_quant`` is False:
        :obj:`float`
            Total value obtained.
    else:
        :obj:`tuple` (:obj:`float`, :obj:`Sequence` of :obj:`float')
            A tuple containing the total value obtained as the first
            element, and a sequence containing the fraction of each
            corresponding item picked up as the second element.

    Notes
    -----
    Uses greedy strategy.
    """
    num_items = len(item_values)

    if item_quantity is None:
        item_quantity = np.ones(num_items)

    if not num_items == len(item_weights) == len(item_quantity):
        raise ValueError(
            "Length of 'item_values' ({}), 'item_weights' ({}) and "
            "'item_quantity' (if given) must be all equal!".format(
                num_items, len(item_weights)))

    value_per_weight = np.zeros(num_items)
    for i in np.arange(num_items):
        if item_weights[i] <= 0:
            raise ValueError(
                "Found negative item weight (index {}.)".format(i))

        value_per_weight[i] = item_values[i] / item_weights[i]

    orig_indices, value_per_weight = zip(
        *sorted(
            zip(np.arange(num_items), value_per_weight),
            key=lambda item: item[0],
            reverse=True))

    cur_weight = 0
    cur_value = 0

    cur_ind = 0
    remaining_lim = knapsack_limit - cur_weight
    quant_per_item = np.zeros(num_items)

    while cur_ind < num_items and remaining_lim > 0:
        sorted_ind = orig_indices[cur_ind]

        item_frac = min(item_quantity[sorted_ind],
                        remaining_lim / item_weights[sorted_ind])

        quant_per_item[sorted_ind] = item_frac

        cur_weight += item_frac * item_weights[sorted_ind]
        cur_value += item_frac * item_values[sorted_ind]

        cur_ind += 1
        remaining_lim = knapsack_limit - cur_weight

    if return_item_quant:
        return cur_value, quant_per_item

    return cur_value


def _test():
    import matplotlib.pyplot as plt

    values = np.array([5, 4, 3, 2, 1])
    weights = np.array([3, 2, 1, 1, 1])

    knapsack_limit = 6

    total_value, quant_per_item = fractional_knapsack(
        knapsack_limit=knapsack_limit,
        item_weights=weights,
        item_values=values,
        item_quantity=np.array([1, 1, 0.75, 1.95, 1]),
        return_item_quant=True)

    assert np.allclose(knapsack_limit, np.sum(quant_per_item * weights))

    plt.title("Weight per item (Total value: {:.2f})".format(total_value))
    plt.bar(np.arange(values.size), height=quant_per_item)
    plt.show()


if __name__ == "__main__":
    _test()
