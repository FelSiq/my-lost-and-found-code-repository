import math
import numpy as np


def check_missingno(seq):
    """Find a missing binary sequence in range [0..n]."""
    ans = ""

    n = len(seq)
    indexes = np.array([i for i in range(n)])

    index_cur_bit = 1
    while len(indexes):
        one_indexes = []

        for index_one_indexes, index_seq in enumerate(indexes):
            if seq[index_seq][-index_cur_bit] == "1":
                one_indexes.append(index_one_indexes)

        if len(one_indexes) < math.ceil(n/2):
            indexes = indexes[one_indexes]
            ans = "1" + ans

        else:
            indexes = np.delete(indexes, one_indexes)
            ans = "0" + ans

        n = len(indexes)
        index_cur_bit += 1

    return ans


if __name__ == "__main__":
    import random
    import sys

    if len(sys.argv) < 2:
        print("usage:", sys.argv[0], "<size> <missing_no>")
        exit(1)

    n = int(sys.argv[1])
    m = int(sys.argv[2])

    fill_size = math.ceil(math.log(n+1, 2))
    seq = [
        format(i, "b").rjust(fill_size, "0") for i in range(n+1)
    ]
    seq.pop(m)
    random.shuffle(seq)

    ans = check_missingno(seq)
    print("answer:", ans, "(" + str(int(ans, 2)) + ")")
