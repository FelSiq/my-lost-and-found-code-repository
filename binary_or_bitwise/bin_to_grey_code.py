"""Convert a binary number to its reflected Gray code."""
import typing as t

import numpy as np


def bin_to_gray(bin_val: str) -> str:
    """Convert a binary value to its reflected gray code."""
    num_vals = np.fromiter(map(int, bin_val), dtype=int)

    gray_val =  [
        1 - num_vals[i] if num_vals[i-1] == 1 else num_vals[i]
        for i in np.arange(num_vals.size - 1, 0, -1)
    ] + [num_vals[0]]

    res = "".join(map(str, gray_val[::-1]))

    return res


def gray_to_bin(gray_val: str) -> str:
    """Convert a reflected gray code to its binary form."""
    num_vals = np.fromiter(map(int, gray_val), dtype=int)

    sigma = np.hstack((0, np.cumsum(num_vals)[:-1])) % 2

    bin_val = [
        1 - num_vals[i] if sigma[i] == 1 else num_vals[i]
        for i in np.arange(num_vals.size - 1, -1, -1)
    ]

    res = "".join(map(str, bin_val[::-1]))

    return res


def _test_01() -> None:
    bin_num_a = "1000"
    print("bin_a:", bin_num_a)

    gray_num = bin_to_gray(bin_num_a)
    print("gray :", gray_num)

    bin_num_b = gray_to_bin(gray_num)
    print("bin_b:", bin_num_b)

    assert bin_num_a == bin_num_b


def _test_02() -> None:
    num_it = 20000
    np.random.seed(16)

    for i in np.arange(num_it):
        num_size = np.random.randint(1, 1000)
        bin_num_a = "".join(np.random.choice(("0", "1"), size=num_size))
        bin_num_b = gray_to_bin(bin_to_gray(bin_num_a))
        assert bin_num_a == bin_num_b
        print("\r{:.2f}".format(100 * i / num_it), end="")


if __name__ == "__main__":
    _test_01()
    _test_02()
