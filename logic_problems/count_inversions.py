"""
	Let be A an array with n distinct numbers.

	Definition: (inversion) if i < j
	and A[i] > A[j] then the pair (i, j)
	is called an "inversion" of A. (Cormen et al)

	This program uses merge sort to calculate the
	number of inversions of a given array with
	O(n * log(n)) time complexity.
"""
import copy
import math


def _count(A, vec_start, vec_middle, vec_end):
    inv_number = 0

    l_vec = A[vec_start:vec_middle] + [math.inf]
    r_vec = A[vec_middle:vec_end] + [math.inf]

    i, j = 0, 0
    subvec_half_size = (vec_end - vec_start) // 2

    for k in range(vec_start, vec_end):
        if l_vec[i] <= r_vec[j]:
            A[k] = l_vec[i]
            i += 1
        else:
            A[k] = r_vec[j]
            if i < subvec_half_size:
                inv_number += subvec_half_size - i
            j += 1

    return inv_number


def _inversions(A, vec_start, vec_end):
    inv_number = 0
    if vec_end - vec_start > 1:
        vec_middle = (vec_start + vec_end) // 2
        inv_number = _inversions(A, vec_start, vec_middle)
        inv_number += _inversions(A, vec_middle, vec_end)
        inv_number += _count(A, vec_start, vec_middle, vec_end)

    return inv_number


def inversions(A):
    _A = copy.copy(A)
    inv_number = _inversions(_A, 0, len(A))
    return inv_number


if __name__ == "__main__":
    import random
    vec = [random.randint(0, 99) for _ in range(5)]
    ans = inversions(vec)
    print("vector =", vec, "answer =", ans)
