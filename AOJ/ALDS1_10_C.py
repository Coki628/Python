# -*- coding: utf-8 -*-

"""
参考：https://atarimae.biz/archives/23930#lm_mn
・連鎖行列積
・行列同士の掛け算
・ほぼ写経。この遷移を自分で書ける気はしない。
"""

import sys
sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

N=INT()
P=[0]*(N+1)
for i in range(N):
    r,c=MAP()
    if i==0:
        P[i]=r
    P[i+1]=c

dp=[[INF]*(N+1) for i in range(N+1)]
for i in range(1, N+1):
    dp[i][i]=0

for l in range(2, N+1):
    for i in range(1, N-l+2):
        j=i+l-1
        for k in range(i, j):
            dp[i][j]=min(dp[i][j], dp[i][k]+dp[k+1][j]+P[i-1]*P[k]*P[j])
print(dp[1][N])
