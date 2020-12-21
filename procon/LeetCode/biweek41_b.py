"""
・自力AC
・中央値
・ひとつ右に動く時、差分を取って、自分より左にいる要素の数だけ増えて、右にいる要素の数だけ減る。
"""

import sys
from typing import List

def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class Solution:
    def getSumAbsoluteDifferences(self, A: List[int]) -> List[int]:
        N = len(A)
        ans = [0] * N
        for i in range(N):
            ans[0] += abs(A[0]-A[i])
        
        for i in range(1, N):
            l = i
            r = N - i
            ans[i] = ans[i-1] + abs(A[i]-A[i-1])*l - abs(A[i]-A[i-1])*r
        return ans

# sol = Solution()
# print(sol.getSumAbsoluteDifferences([1,4,6,8,10]))
