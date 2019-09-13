def sort_stack(s):
    """Sort a stack using only another stack as helper.

    Time Complexity: O(n**2)
    Space Complexity: Theta(1)
    """
    temp = []

    while s:
        if not temp or s[-1] >= temp[-1]:
            temp.append(s.pop())

        else:
            aux = s.pop()
            s.append(temp.pop())
            s.append(aux)

    while temp:
        s.append(temp.pop())

    return s


if __name__ == "__main__":
    import random

    def is_sorted(s):
        return s == sorted(s, reverse=True)

    random.seed(1234)
    
    for i in range(2000):
        s = [
          random.randint(-1000, 1000)
          for _ in range(random.randint(1, 30))
        ]
    
        assert is_sorted(sort_stack(s))
