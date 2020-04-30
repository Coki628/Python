"""
・LeetCode初提出。書式が独特なのね。
"""

from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        N = len(nums)
        for i in range(N):
            for j in range(i+1, N):
                if nums[i] + nums[j] == target:
                    return [i, j]
