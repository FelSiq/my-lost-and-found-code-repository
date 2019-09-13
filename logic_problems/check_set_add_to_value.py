"""
	Given a set A with n >= 0 elements,
	check if two values of A add up to a
	given value c with theta(n*logn) com-
	plexity.
"""


def find_add(s, val):
    """Check if two elements of set s add up to val.

	Return tuple with values of s if found,
	or (None, None) otherwise.
	"""
    # Cost: n*log(n)
    sorted_set = sorted(list(s))

    i, j = 0, len(sorted_set) - 1
    while i < j:
        add = sorted_set[i] + sorted_set[j]
        if add == val:
            return sorted_set[i], sorted_set[j]

        elif add > val:
            j -= 1

        else:
            i += 1

    return None, None


if __name__ == "__main__":
    import sys
    import random

    if len(sys.argv) < 2:
        print(
            "usage:",
            sys.argv[0],
            "<value>",
            "[--set set_values, default is random]",
            "[--sep, default is \" \"]",
            sep="\n\t")
        exit(1)

    try:
        val = int(sys.argv[1])
    except:
        print("value must be a integer.")
        exit(2)

    try:
        sep = sys.argv[1 + sys.argv.index("--sep")]
    except:
        sep = " "

    try:
        s = set(map(int, sys.argv[1 + sys.argv.index("--set")].split(sep)))
    except:
        s = set([random.randint(-25, 25) for _ in range(25)])

    a, b = find_add(s, val)

    print("set:", s)
    if a and b:
        print("answer:", a, "+", b)
    else:
        print("no answer.")
