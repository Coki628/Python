# -*- coding: utf-8 -*-

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

S=input()
N=len(S)
YAHOO='yahoo'

dp=[0]*(N+1)
for i in range(min(N+1, 6)):
    dp[i]=min(5-LCS(S[:i], YAHOO), i)

for i in range(6, N+1):
    s=S[i-5:i]
    cnt=5-LCS(s, YAHOO)
    dp[i]=min(dp[i-5]+cnt, dp[i-1]+1)

for i in range(max(N-4, 0), N+1):
    s=S[i:N+1]
    cnt=5-LCS(s, YAHOO)
    dp[N]=min(dp[N], dp[i]+cnt)

print(dp[N])
