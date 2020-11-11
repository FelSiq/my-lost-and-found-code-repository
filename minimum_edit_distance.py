"""
insert
delete
replace
swap?
"""


def med_rec(word_a: str, word_b: str) -> int:
    def rec(i, j):
        if i >= len(word_a) or j >= len(word_b):
            return len(word_a) - i + len(word_b) - j

        if memo[i][j] >= 0:
            return memo[i][j]

        if word_a[i] == word_b[j]:
            ret = rec(i + 1, j + 1)

        else:
            ret = 1 + min(rec(i + 1, j), rec(i, j + 1))

        memo[i][j] = ret

        return ret

    memo = [(1 + len(word_b)) * [-1] for _ in range(1 + len(word_a))]

    return rec(0, 0)


def med(word_a: str, word_b: str) -> int:
    memo = [(1 + len(word_b)) * [0] for _ in range(2)]

    for i in range(len(word_a) + 1):
        im = i % 2
        imm = (i - 1) % 2

        for j in range(len(word_b) + 1):
            if i == 0 or j == 0:
                memo[im][j] = max(i, j)

            elif word_a[i - 1] == word_b[j - 1]:
                memo[im][j] = memo[imm][j - 1]

            else:
                memo[im][j] = 1 + min(memo[imm][j], memo[im][j - 1])

    return memo[len(word_a) % 2][-1]


def _test():
    assert med("abc", "abcd") == 1
    assert med("abc", "abdc") == 1
    assert med("abc", "dabc") == 1
    assert med("abc", "adbc") == 1
    assert med("abc", "adabc") == 2
    assert med("dabc", "adabc") == 1
    assert med("abc", "acb") == 1
    print("ok")

    import random

    random.seed(128)

    for i in range(500):
        print(i, end="... ")
        size_a = random.randint(0, 40)
        size_b = random.randint(0, 40)
        str_a = "".join(
            [chr(random.randint(ord("a"), 1 + ord("z"))) for _ in range(size_a)]
        )
        str_b = "".join(
            [chr(random.randint(ord("a"), 1 + ord("z"))) for _ in range(size_b)]
        )
        assert med_rec(str_a, str_b) == med(str_a, str_b), f"{str_a}, {str_b}"
        print("ok")


if __name__ == "__main__":
    _test()
