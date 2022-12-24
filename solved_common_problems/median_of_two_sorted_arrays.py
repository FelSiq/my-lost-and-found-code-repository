# based on: https://leetcode.com/explore/interview/card/google/63/sorting-and-searching-4/3080/discuss/2651020/C++-oror-SOLUTION
class findMedianSortedArrays:
    def find(self, nums1: list[int], nums2: list[int]) -> float:
        n1 = len(nums1)
        n2 = len(nums2)
        n = n1 + n2

        if n1 > n2:
            nums1, nums2 = nums2, nums1
            n1, n2 = n2, n1

        partition = (n + 1) // 2

        if n1 == 0:
            return (
                nums2[n2 // 2]
                if n2 % 2
                else 0.5 * (nums2[n2 // 2] + nums2[(n2 - 1) // 2])
            )

        left1, right1 = 0, n1
        while left1 <= right1:
            k1 = (left1 + right1) // 2
            k2 = partition - k1
            l1 = nums1[k1 - 1] if k1 >= 1 else float("-inf")
            l2 = nums2[k2 - 1] if k2 >= 1 else float("-inf")
            r1 = nums1[k1] if k1 < n1 else float("+inf")
            r2 = nums2[k2] if k2 < n2 else float("+inf")

            if l1 <= r2 and l2 <= r1:
                return max(l1, l2) if n % 2 else 0.5 * (max(l1, l2) + min(r1, r2))
            if l1 > l2:
                right1 = k1 - 1
            else:
                left1 = k1 + 1

        return 0.0
