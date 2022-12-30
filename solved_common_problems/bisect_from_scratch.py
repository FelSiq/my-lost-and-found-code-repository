import operator

def bisect(arr, x, /, *, left: bool = True) -> int:
    i = 0
    j = len(arr)
    fn_op = operator.lt if left else operator.le

    while i < j:
        middle = i + (j - i) // 2
        if fn_op(arr[middle], x):
            i = middle + 1
        else:
            j = middle
        
    return i


if __name__ == "__main__":
    import random
    import bisect as bisect_builtin

    random.seed(16)

    for _ in range(10000):
        n = random.randint(1, 100)
        x = random.randint(-12, 12)
        arr = sorted([random.randint(-11, 11) for _ in range(n)])

        aux_r_a = bisect(arr, x, left=False)
        aux_r_b = bisect_builtin.bisect_right(arr, x)
        assert aux_r_a == aux_r_b, (aux_r_a, aux_r_b)

        aux_l_a = bisect(arr, x, left=True)
        aux_l_b = bisect_builtin.bisect_left(arr, x)
        assert aux_l_a == aux_l_b, (aux_l_a, aux_l_b)
