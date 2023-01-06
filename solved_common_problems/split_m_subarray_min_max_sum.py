def split_into_m_subarray_min_max_sum(nums: list[int], k: int) -> int:
    def split_max_m(m):
        splits = 1
        cumsum = 0
        
        for v in nums:
            if cumsum + v > m:
                splits += 1
                cumsum = 0
                
            cumsum += v
                
        return splits
    
    i, j = max(nums), sum(nums)
    res = 0
    
    while i <= j:
        guess = i + (j - i) // 2
    
        if split_max_m(guess) <= k:
            j = guess - 1
            res = guess
        
        else:
            i = guess + 1
    
    return res
