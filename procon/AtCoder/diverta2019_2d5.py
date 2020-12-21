# -*- coding: utf-8 -*-

"""
・個数制限なしの場合、同じ配列を更新していけばいいので、
　ループ入れ替えて配列を1次元に。
・計算量はN^2の2500万と定数3で7500万、pypyAC1.9秒。
"""

import sys
from itertools import accumulate

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()
B = LIST()

def solve(N):
    dp = [0] * (N+1)
    for i in range(N+1):
        dp[i] = i
    for i in range(N+1):
        for j in range(3):
            a, b = A[j], B[j]
            if i+a <= N:
                dp[i+a] = max(dp[i+a], dp[i] + b)
    return dp[N]

N2 = solve(N)
A, B = B, A
ans = solve(N2)
print(ans)
