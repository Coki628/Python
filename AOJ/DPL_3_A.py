# -*- coding: utf-8 -*-

"""
・最大正方形
・2方向から遷移してくるLCS系のDP
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

H,W=MAP()
# 四方に一回り大きいグリッドを作る
grid = list2d(H+2, W+2, 0)
for i in range(1, H+1):
    row = LIST()
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

# dp[i][j] := i,jを右端とする最大正方形の辺の長さ
dp=list2d(H+2, W+2, 0)
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j]!=1:
            # i,jが汚れマスでなければ、左、左上、上から遷移させてくる
            dp[i][j]=min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])+1
mx=0
for i in range(1, H+1):
    mx=max(mx, max(dp[i]))
print(mx**2)
