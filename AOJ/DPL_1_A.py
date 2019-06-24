# -*- coding: utf-8 -*-

"""
・コイン問題
・1次元minmaxDP+内ループで配る。貰ってもいい。
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

N,M=MAP()
C=LIST()

dp=[INF]*(N+1)
dp[0]=0
for i in range(N):
    for j in range(M):
        if i+C[j]<=N:
            dp[i+C[j]]=min(dp[i+C[j]], dp[i]+1)
print(dp[N])
