# -*- coding: utf-8 -*-

"""
・自力WA。。
"""

import sys

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
A = [0] + LIST()

# dp[i] := i番目をスローロリスにするか選べる状態で、その後ろをスローロリスにできる最大値
dp = [0] * (N+1)
# 後ろから見ていく
for i in range(N, 0, -1):
    a = A[i]
    # i番目をスローロリスにしない
    dp[i-1] = max(dp[i-1], dp[i])
    # i番目をスローロリスにする(i-a番目まではできなくなる)
    dp[i-a-1] = max(dp[i-a-1], dp[i] + 1)
print(dp[0])
