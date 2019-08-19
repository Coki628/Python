# -*- coding: utf-8 -*-

"""
・配るDP、minmax系、たくさん遷移する系
・愚直実装を遷移の見やすさのために添字逆にした。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
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

N,X=MAP()
# 1-indexedに揃える
T=[0]+[INT() for i in range(N)]

# dp[i][j] := j個目の時刻に、i回起動した状態の知力の最大値
dp=list2d(N+1, N+1, 0)
for j in range(1, N+1):
    # 初期化：開始時スタミナ満タンなので、1回起動なら確実にX使える
    dp[1][j]=X

for i in range(1, N):
    for j in range(i, N):
        for k in range(j+1, N+1):
            dp[i+1][k]=max(dp[i+1][k], dp[i][j]+min(X, T[k]-T[j]))

# 時刻Nまで終わって、各i回起動した状態を出力
[print(dp[i][N]) for i in range(1, N+1)]
