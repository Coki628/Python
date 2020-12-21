# -*- coding: utf-8 -*-

"""
・さくっと自力AC！
・tkppc4day1iとほぼ同じ感じで解けた。
・貰うDP、数え上げDP
・累積和で前計算して、貰う遷移を高速化。
・該当位置の取得には二分探索。
・公式解はちょっと違ったけど、まあこれでもいいよね。
"""

import sys
from itertools import accumulate
from bisect import bisect_right

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = [INT() for i in range(N)]

A.sort()
dp = list2d(4, N, 0)
for i in range(N):
    dp[0][i] = 1
dp[0] = list(accumulate(dp[0]))

for i in range(1, 4):
    for j in range(N):
        idx = bisect_right(A, A[j]//2) - 1
        if idx != -1:
            dp[i][j] = dp[i-1][idx]
            dp[i][j] %= MOD
    dp[i] = list(accumulate(dp[i]))
print(dp[3][-1]%MOD)
