# -*- coding: utf-8 -*-

"""
・自力AC！
・1次元+フラグの2次元DP
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, C = MAP()
A = LIST()
B = LIST()

# dp[i][j] := 直前にエレベーターで来ている(i=1)or来ていない(i=0)の状態でのj階到達の最小コスト
dp = list2d(2, N, INF)
dp[0][0] = 0
for i in range(N-1):
    dp[0][i+1] = min(dp[0][i] + A[i], dp[1][i] + A[i])
    dp[1][i+1] = min(dp[0][i] + B[i] + C, dp[1][i] + B[i])
ans = [0] * N
for i in range(N):
    ans[i] = min(dp[0][i], dp[1][i])
print(*ans)
