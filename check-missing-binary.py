import math
import numpy as np


def check_missingno(seq):
    ans = ""

    n = len(seq)
    indexes = np.array([i for i in range(n)])

    j = 1
    while len(indexes):
        zero_indexes = []
        one_indexes = []

        for k, i in enumerate(indexes):
            if seq[i][-j] == "1":
                one_indexes.append(k)
            else:
                zero_indexes.append(k)

        if len(one_indexes) < math.ceil(n/2):
            indexes = np.delete(indexes, zero_indexes)
            ans = "1" + ans
        else:
            indexes = np.delete(indexes, one_indexes)
            ans = "0" + ans

        n = len(indexes)
        j += 1

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
