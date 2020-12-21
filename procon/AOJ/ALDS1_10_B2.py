# -*- coding: utf-8 -*-

"""
・DP基礎
・LCS
・AOJだとpypy使えなくて1000^2*20クエリ=2000万がしんどかったからC++でAC
・やっぱりLCSは配るより貰うDPのがいいかな、とこっちにしてみた。
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
    for i in range(1, N+1):
        for j in range(1, M+1):
            # 上から受け取る
            dp[i][j] = max(dp[i][j], dp[i-1][j])
            # 左から受け取る
            dp[i][j] = max(dp[i][j], dp[i][j-1])
            # 今回増やす文字が一致するなら、LCSが1文字増える
            if A[i-1] == B[j-1]:
                # 左上から受け取る
                dp[i][j] = max(dp[i][j], dp[i-1][j-1]+1)
    return dp[N][M]

Q=INT()

for _ in range(Q):
    a=input()
    b=input()
    print(LCS(a, b))
