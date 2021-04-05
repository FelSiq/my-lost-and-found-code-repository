"""Activity selector problem using greedy strategy.

The problem can be proposed as follows:
Given the Start and Finish time of each available activity,
maximize the number of non-overlapping activies.
"""
import typing as t
import warnings

import matplotlib.pyplot as plt
import numpy as np


def _plot_schedule(chosen_act_inds: t.Iterable[int],
                   time_start: t.Sequence[t.Union[int, float]],
                   time_end: t.Sequence[t.Union[int, float]],
                   activity_names: t.Optional[t.Sequence[str]] = None) -> None:
    """Plot the found schedule."""
    num_act = len(time_end)

    plt.title("Found activity schedule")
    plt.ylabel("Activity index")
    plt.xlabel("Time")

    if activity_names is not None and len(activity_names) != num_act:
        warnings.warn(
            "'activity_names' length ({}) does not match with the "
            "number of activities ({}.)".format(len(activity_names), num_act),
            UserWarning)
        activity_names = None

    plt.yticks(np.arange(num_act), labels=activity_names)

    for ind in np.arange(num_act):
        i_is_chosen = ind in chosen_act_inds

        plt.hlines(
            y=ind,
            xmin=time_start[ind],
            xmax=time_end[ind],
            color="red" if i_is_chosen else "black",
            linestyle="-" if i_is_chosen else "--")

        if i_is_chosen:
            plt.vlines(
                x=[time_start[ind], time_end[ind]],
                ymin=0,
                ymax=num_act - 1,
                color="gray",
                linestyle=":")

    plt.show()


def activity_selector(time_start: t.Sequence[t.Union[float, int]],
                      time_end: t.Sequence[t.Union[int, float]],
                      sort_time_end: bool = True,
                      activity_names: t.Optional[t.Sequence[str]] = None,
                      plot: bool = True) -> t.Set[int]:
    """Returns the maximum-sized set of non-overllaping activities.

    Arguments
    ---------
    time_start : :obj:`t.Sequence` of :obj:`float` or :obj:`int`
        Sequence of numbers such that the $i$th index represents
        the the time that the $i$th activity starts.

    time_end : :obj:`t.Sequence` of :obj:`float` or :obj:`int`
        Same as above, but with the ending time of the activities.
        If given sorted in increasingly order, then you can set
        ``set_time_end`` to False and improve the execution time.

    sort_time_end : :obj:`bool`, optional
        If True, then sort the activies (O(n * lg(n)) time) by the
        ending time in increasing order. This aggregates the cost of
        sorting and also the cost of translating the indices of the
        output to the indices of the original given arrays (O(n)).
        If the ``time_end`` sequence is already given sorted, then
        you can set this argument to False.

    activity_names : :obj:`t.Sequence` of :obj:`str`, optional
        Name of each index-correspondent activity. Is used only if
        ``plot`` is True. Ignored otherwise.

    plot : :obj:`bool`, optional
        If True, then plot the results.

    Returns
    -------
    :obj:`set` of :obj:`int`
        Set of indexes corresponding to the selected activities.

    Notes
    -----
    Uses greedy strategy.
    """
    if len(time_start) != len(time_end):
        raise ValueError("Lengths of 'time_start' and 'time_end' must match!"
                         "(Got {} and {}.)".format(
                             len(time_start), len(time_end)))

    if sort_time_end:
        sorted_t_end, sorted_t_start = zip(
            *sorted(zip(time_end, time_start), key=lambda item: item[0]))

        _index_mapping = {
            (t_pair[0], t_pair[1]): i
            for i, t_pair in enumerate(zip(time_start, time_end))
        }

    else:
        sorted_t_end, sorted_t_start = time_end, time_start  # type: ignore

    num_act = len(time_end)

    chosen_act_inds = {0}
    last_chosen_ind = 0

    for ind in np.arange(1, num_act):
        if sorted_t_start[ind] >= sorted_t_end[last_chosen_ind]:
            chosen_act_inds.add(ind)
            last_chosen_ind = ind

    if sort_time_end:
        chosen_act_inds = {
            _index_mapping[(sorted_t_start[ind], sorted_t_end[ind])]
            for ind in chosen_act_inds
        }

    if plot:
        _plot_schedule(
            chosen_act_inds=chosen_act_inds,
            time_start=time_start,
            time_end=time_end,
            activity_names=activity_names)

    return chosen_act_inds


def _test() -> None:
    """Testing with the Cormen et al. example (page 415, 3rd ed.)"""
    time_start = [1, 3, 0, 5, 3, 5, 6, 8, 8, 2, 12]
    time_end = [4, 5, 6, 7, 9, 9, 10, 11, 12, 14, 16]
    activity_names = [
        "Activity {}".format(chr(i + ord('a')))
        for i in np.arange(len(time_start))
    ]

    np.random.seed(16)

    _aux = np.vstack((time_start, time_end, activity_names)).T
    np.random.shuffle(_aux)
    time_start, time_end, activity_names = _aux[:, 0].astype(
        int), _aux[:, 1].astype(int), _aux[:, 2]

    chosen_acts_ind = activity_selector(
        time_start=time_start,
        time_end=time_end,
        sort_time_end=True,
        activity_names=activity_names,
        plot=True)

    print("Schedule:")
    for ind in chosen_acts_ind:
        print("{} | Start: {} - End: {}".format(
            activity_names[ind], time_start[ind], time_end[ind]))


if __name__ == "__main__":
    _test()
