"""
・二分探索
・真ん中をどう抜くか、になるので、前半で単調増加のところと後半で単調増加のところだけ持ってきて、
　前半を全探索して、その前半が使える後半の位置をにぶたんする。
・前半を1個も使わないパターンを見落としててWA。
　りとこ優しくて、テストケース見えちゃったから分かったけど、自力で見つけ出せただろうか。
"""

from typing import List

import sys
from bisect import bisect_left

def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class Solution:
    def findLengthOfShortestSubarray(self, arr: List[int]) -> int:
        N = len(arr)

        A1 = []
        for i in range(N-1):
            if arr[i] > arr[i+1]:
                A1 = arr[:i+1]
                break
        A2 = []
        for i in range(N-2, -1, -1):
            if arr[i] > arr[i+1]:
                A2 = arr[i+1:]
                break
        if not A1:
            return 0

        # 前半0個を考慮するため番兵を仕込む
        A1.insert(0, 0)
        N1 = len(A1)
        N2 = len(A2)
        ans = INF
        for i in range(N1):
            lcnt = i
            rcnt = N2 - bisect_left(A2, A1[i])
            ans = min(ans, N - (lcnt+rcnt))
        return ans

# sol = Solution()
# print(sol.findLengthOfShortestSubarray(arr))
