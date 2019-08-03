# -*- coding: utf-8 -*-

"""
・文字列の編集距離(レーベンシュタイン距離)
・LCSみたいな2方向DP
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

S=input()
N=len(S)
T=input()
M=len(T)

# dp[i][j] := T[0..j]をS[0..i]にするのにかかる最小操作数
dp = [[INF] * (M+1) for i in range(N+1)]
# 初期化(Sが0文字でTも0文字なら何もしなくていい)
dp[0][0] = 0
for i in range(N+1):
    for j in range(M+1):
        # 挿入
        if i < N:
            dp[i+1][j] = min(dp[i+1][j], dp[i][j]+1)
        # 削除
        if j < M:
            dp[i][j+1] = min(dp[i][j+1], dp[i][j]+1)
        # 置換
        if i < N and j < M:
            dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j]+1)
            if S[i] == T[j]:
                # 今回増やす文字が一致するなら、ノーコストで遷移できる
                dp[i+1][j+1] = min(dp[i+1][j+1], dp[i][j])
# SがN文字、TがM文字で終わる場合の最小回数
print(dp[N][M])
