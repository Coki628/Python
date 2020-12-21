# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc027
・公式の部分点解法
・普通のminmaxDP、配るDP
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

S=input()
N=len(S)

M=S.count('M')
A=[0]*(M+1)
mcnt=0
for s in S:
    if s=='M':
        mcnt+=1
    elif s=='+':
        A[mcnt]+=1
    else:
        A[mcnt]-=1

# dp[i][j] := i回目の移動後、座標jにいる時の最大値
dp=list2d(M+1, N*2+1, -INF)

# 負の数を考慮するため、Nを座標0とする
dp[0][N]=0
for i in range(M):
    for j in range(N*2+1):
        if dp[i][j]!=-INF:
            dp[i+1][j-1]=max(dp[i+1][j-1], dp[i][j]+A[i+1]*(j-N-1))
            dp[i+1][j+1]=max(dp[i+1][j+1], dp[i][j]+A[i+1]*(j-N+1))
print(dp[M][N])
