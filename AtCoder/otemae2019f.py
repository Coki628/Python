# -*- coding: utf-8 -*-

"""
・部分点用
・制約2000で3重ループのDPしてるので、これはTLE想定。
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

N, D = MAP()
A = [[]] * D
for i in range(D):
    A[i] = [0] + list(accumulate(LIST()))

# dp[i][j] := i日目まで見て、コインをj枚右に移動させた状態での最小コスト
dp = list2d(D+1, N+1, INF)
dp[0][0] = 0
for i in range(D):
    for j in range(N+1):
        for k in range(j, N+1):
            dp[i+1][k] =  min(dp[i+1][k], dp[i][j] + abs((A[i][N] - A[i][k]) - (A[i][k] - A[i][0])))
ans = min(dp[D])
print(ans)
