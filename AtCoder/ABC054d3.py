# -*- coding: utf-8 -*-

"""
・DP
・うまくいかなかった。
"""
 
import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7
 
N, Ma, Mb = map(int, input().split())
l = [[0] * 3 for i in range(N)]
for i in range(N):
    l[i][0], l[i][1], l[i][2] = map(int, input().split())
 
dp = [[[0] * 401  for i in range(401)] for i in range(N+1)]
dp[0][0][0] = 0
for i in range(N):
    for j in range(401):
        for k in range(401):
            dp[i+1][j][k] = dp[i][j][k]
            dp[i+1][j+l[i][0]][k+l[i][1]] = dp[i][j][k] + l[i][2]

print()
