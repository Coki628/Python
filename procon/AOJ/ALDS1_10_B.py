# -*- coding: utf-8 -*-

"""
・DP基礎
・LCS
・AOJだとpypy使えなくて1000^2*20クエリ=2000万がしんどかったからC++でAC
"""

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

def LCS(A, B):
    N = len(A)
    M = len(B)
    dp = [[0] * (M+1) for i in range(N+1)]
    for i in range(N+1):
        for j in range(M+1):
            # 下に送る
            if i < N:
                dp[i+1][j] = max(dp[i+1][j], dp[i][j])
            # 右に送る
            if j < M:
                dp[i][j+1] = max(dp[i][j+1], dp[i][j])
            # 今回増やす文字が一致するなら、LCSが1文字増える
            if i < N and j < M and A[i] == B[j]:
                # 右下に送る
                dp[i+1][j+1] = max(dp[i+1][j+1], dp[i][j]+1)
    return dp[N][M]

Q=INT()

for _ in range(Q):
    a=input()
    b=input()
    print(LCS(a, b))
