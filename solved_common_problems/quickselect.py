import random


class FindKthElement:
    def partition(self, left, right, pivot_index):
        pivot = self.nums[pivot_index]
        self.nums[pivot_index], self.nums[right] = (
            self.nums[right],
            self.nums[pivot_index],
        )

        store_index = left
        for i in range(left, right):
            if self.nums[i] < pivot:
                self.nums[i], self.nums[store_index] = (
                    self.nums[store_index],
                    self.nums[i],
                )
                store_index += 1

        self.nums[right], self.nums[store_index] = (
            self.nums[store_index],
            self.nums[right],
        )
        return store_index

    def select(self, left, right, k_smallest):
        if left == right:
            return self.nums[left]

        pivot_index = random.randint(left, right)
        pivot_index = self.partition(left, right, pivot_index)

        if k_smallest == pivot_index:
            return self.nums[k_smallest]
        if k_smallest < pivot_index:
            return self.select(left, pivot_index - 1, k_smallest)
        return self.select(pivot_index + 1, right, k_smallest)

    def find_kth_largest(self, nums, k):
        assert k >= 1
        self.nums = list(nums)
        return self.select(0, len(nums) - 1, len(nums) - k)

    def find_kth_smallest(self, nums, k):
        assert k >= 1
        self.nums = list(nums)
        return self.select(0, len(nums) - 1, k - 1)


if __name__ == "__main__":
    finder = FindKthElement()
    random.seed(24)

    for i in range(3000):
        k, n = random.randint(1, 100), random.randint(0, 1000)
        n += k

        nums = [random.randint(-100, 100) + random.random() for _ in range(n)]
        sorted_nums = sorted(nums)

        kth_smallest = finder.find_kth_smallest(nums, k)
        kth_largest = finder.find_kth_largest(nums, k)

        assert sorted_nums[k - 1] == kth_smallest
        assert sorted_nums[n - k] == kth_largest
