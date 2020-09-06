"""
"""

from typing import List

import sys

def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class Solution:
    def numWays(self, S: str) -> int:
        N = len(S)

        # 3つに分けれない、と全部0を例外処理
        cnt1 = S.count('1')
        if cnt1 % 3 != 0:
            return 0
        if cnt1 == 0:
            return (N-1)*(N-2) // 2 % MOD

        each = cnt1 // 3
        cnt1 = 0
        sm1 = sm2 = 1
        for s in S:
            # 1つ目の区切りとして使える
            if s == '0' and cnt1 == each:
                sm1 += 1
            # 2つ目の区切りとして使える
            if s == '0' and cnt1 == each*2:
                sm2 += 1
            if s == '1':
                cnt1 += 1
        ans = sm1 * sm2 % MOD
        return ans

# sol = Solution()
# print(sol.numWays(s))
