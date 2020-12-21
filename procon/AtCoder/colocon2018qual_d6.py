# -*- coding: utf-8 -*-

"""
・配るDP、minmax系、たくさん遷移する系
・numpy実装、まあこれでもTLEだけども。
"""

import sys
import numpy as np

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

N, X = MAP()
# 1-indexedに揃える
T = np.zeros(N+1, dtype=np.int64)
T[1:] = np.array([INT() for i in range(N)], dtype=np.int64)

# dp[i][j] := j個目の時刻に、i回起動した状態の知力の最大値
dp = np.zeros((N+1, N+1), dtype=np.int64)
for j in range(1, N+1):
    # 初期化：開始時スタミナ満タンなので、1回起動なら確実にX使える
    dp[1,j] = X

for i in range(1, N):
    # for j in range(i, N):
    for k in range(1, N+1):
        # 各k個目の時刻への遷移をまとめてやる
        dp[i+1,i+k:N+1] = np.maximum(
            dp[i+1,i+k:N+1], 
            dp[i,i:N+1-k]+np.minimum(X, T[i+k:N+1]-T[i:N+1-k])
        )
        # dp[i+1][k]=max(dp[i+1][k], dp[i][j]+min(X, T[k]-T[j]))

# 時刻Nまで終わって、各i回起動した状態を出力
[print(dp[i,N]) for i in range(1, N+1)]
