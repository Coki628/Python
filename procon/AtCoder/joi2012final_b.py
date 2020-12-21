# -*- coding: utf-8 -*-

"""
・TLE。400万、3次元DPだとpypyでも1秒はきつい。
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
S = input()
T = input()

# dp[i][j][k] := Sをi文字目、Tをj文字目まで見て、最後に使ったのがI,O(k=0,1)の時の最大の長さ
dp = list3d(N+1, M+1, 2, -INF)
# 開始地点としてIしか使えないように、直前がOの所だけ初期化
for i in range(N+1):
    for j in range(M+1):
        dp[i][j][1] = 0
for i in range(N+1):
    for j in range(M+1):
        if i < N:
            # Sから1文字使う
            if S[i] == 'I':
                dp[i+1][j][0] = max(dp[i+1][j][0], dp[i][j][1] + 1)
            else:
                dp[i+1][j][1] = max(dp[i+1][j][1], dp[i][j][0] + 1)
        if j < M:
            # Tから1文字使う
            if T[j] == 'I':
                dp[i][j+1][0] = max(dp[i][j+1][0], dp[i][j][1] + 1)
            else:
                dp[i][j+1][1] = max(dp[i][j+1][1], dp[i][j][0] + 1)
ans = 0
for i in range(N+1):
    for j in range(M+1):
        # 終了地点がI(直前がI)のものから最大を取る
        ans = max(ans, dp[i][j][0])
print(ans)
