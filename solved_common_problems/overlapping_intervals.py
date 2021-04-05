"""Check whether a new interval overlaps existing ones."""
import bisect


class NonOverlappingInts:
    def __init__(self):
        self.intervals = []

    def __repr__(self):
        return str(self.intervals)

    def __getitem__(self, i):
        return self.intervals[i]

    def __len__(self):
        return len(self.intervals)

    def __contains__(self, interval):
        start, end = interval
        return self.non_overlapping_ind(start, end) < 0

    def __iter__(self):
        return iter(self.intervals)

    @staticmethod
    def is_overlapping(start_a, end_a, start_b, end_b):
        return start_a <= end_b and end_a >= start_b

    def non_overlapping_ind(self, start, end):
        ind = bisect.bisect_left(self.intervals, (start, end))

        if ind >= 1:
            start_prev, end_prev = self.intervals[ind - 1]
            if self.is_overlapping(start, end, start_prev, end_prev):
                return -1

        if ind < len(self.intervals):
            start_cur, end_cur = self.intervals[ind]
            if self.is_overlapping(start, end, start_cur, end_cur):
                return -1

        return ind

    def add(self, start, end):
        ind = self.non_overlapping_ind(start, end)

        if ind < 0:
            return False

        self.intervals.insert(ind, (start, end))

        return True


def _test():
    import numpy as np

    num_tests = 1000
    np.random.seed(64)

    for i in np.arange(num_tests):
        print(f"Test {i} ...")
        intervals = NonOverlappingInts()
        num_intervals = np.random.randint(50)

        for a, d in np.random.randint(1000, size=(num_intervals, 2)):
            start, end = a, a + d // 6

            check_a = (start, end) in intervals
            check_b = intervals.add(start, end)

            assert check_a == (not check_b)

            if check_b:
                print("      Added:", (start, end))
            else:
                print("    Refused:", (start, end))

        assert sorted(intervals.intervals) == intervals.intervals

        for i in range(len(intervals) - 1):
            _, end_cur = intervals[i]
            start_next, _ = intervals[i + 1]
            assert end_cur < start_next

        print(intervals)
        print("To be added :", num_intervals)
        print("Really added:", len(intervals))


if __name__ == "__main__":
    _test()
