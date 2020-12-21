# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・いわゆる普通のナップザック系2次元DP
・計算量O(NM)で100万、pythonAC1.0秒。
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

N, M = MAP()
A = LIST(N) + [0]
B = LIST(M)

# dp[i][j] := i日目に、都市jにいる状態の、疲労度の最小値
dp = list2d(M+1, N+1, INF)
dp[0][0] = 0
for i, b in enumerate(B):
    for j, a in enumerate(A):
        dp[i+1][j] = min(dp[i+1][j], dp[i][j])
        if j+1 <= N:
            dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j] + a * b)
ans = dp[M][N]
print(ans)
