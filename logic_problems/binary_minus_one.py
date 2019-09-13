"""Ugly code to compute N - 1 using boolean logic."""
import sys

import numpy as np

n = None
if len(sys.argv) > 1:
    n = bin(int(sys.argv[1]))[2:]


def minus_one(n: str) -> str:
    ans = []

    n = n[::-1]

    if n == "0":
        raise ValueError("'n' must be positive.")

    for bit_index, bit_value in enumerate(n):
        cur_bit = bit_value == "1"
        prev_bits = list(map(int, n[:bit_index]))

        aux1 = np.prod(np.logical_not(prev_bits + [cur_bit]))
        aux2 = cur_bit * (sum(prev_bits) > 0)
        ans.append(int(np.logical_or(aux1, aux2)))

    return "".join(map(str, ans[::-1]))


if n is None:
    for i in range(100):
        num = np.random.randint(1, 9999)
        num_minus_one = minus_one(bin(num)[2:])
        assert num - 1 == int(num_minus_one, 2)

else:
    print(minus_one(n))
