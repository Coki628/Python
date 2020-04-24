# -*- coding: utf-8 -*-

"""
・pythonACしないかなー、と枝刈りとか試みるもダメ。
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

N, K = MAP()
M = N - K
A = [0] + LIST()

# dp[i][j] := 列をi個使って、最後に使ったのがj列目の時の最小コスト
dp = list2d(M+1, N+1, INF)
dp[0][0] = 0
for i in range(M):
    for j, h1 in enumerate(A[i:], i):
        if dp[i][j] == INF:
            break
        for k in range(j+1, N+1):
            h2 = A[k]
            dp[i+1][k] = min(dp[i+1][k], dp[i][j] + max(h2 - h1, 0))
print(min(dp[M]))
