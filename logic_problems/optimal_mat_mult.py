import typing as t

import numpy as np


def matrix_chain_order(
        dims: t.Sequence[int],
        verbose: bool = False) -> t.Tuple[np.ndarray, np.ndarray]:
    """Find the optimal way of multiplying n matrices with DP."""
    def _traceback(sol_table: np.ndarray, i: int, j: int):
        if i == j:
            return "M_{}".format(i)

        else:
            return "({} * {})".format(
                        _traceback(sol_table, i, sol_table[i, j-1]),
                        _traceback(sol_table, sol_table[i, j-1] + 1, j))

    mat_num = len(dims) - 1

    memo_table = np.zeros((mat_num, mat_num))
    sol_table = np.full((mat_num - 1, mat_num - 1), -1, dtype=int)

    for chain_len in np.arange(2, mat_num + 1):
        for ind_start in np.arange(mat_num - chain_len + 1):
            ind_end = ind_start + chain_len - 1
            memo_table[ind_start, ind_end] = np.inf

            for ind_cut in np.arange(ind_start, ind_end + 1):

                subproblems_cost = (memo_table[ind_start, ind_cut - 1] +
                                    memo_table[ind_cut, ind_end])

                cur_mat_mult_cost = (
                    dims[ind_start] * dims[ind_cut] * dims[ind_end + 1])

                total_cur_cost = subproblems_cost + cur_mat_mult_cost

                if total_cur_cost < memo_table[ind_start, ind_end]:
                    memo_table[ind_start, ind_end] = total_cur_cost
                    sol_table[ind_start, ind_end - 1] = ind_cut - 1

    if verbose:
        print("Memoization table:\n{}\n\nSolution table:\n{}\n".format(memo_table, sol_table))

    return _traceback(sol_table, 0, mat_num-1)


if __name__ == "__main__":
    DIMS = [30, 35, 15, 5, 10, 20, 25]
    DIMS = [5, 10, 3, 12, 5, 50, 6]
    solution = matrix_chain_order(DIMS, verbose=True)
    print("Final soluion:", solution)
