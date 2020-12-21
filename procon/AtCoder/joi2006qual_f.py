# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-13、さくっと自力AC
・グリッドDP
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
INF = float('inf')
MOD = 10 ** 9 + 7

directions = ((1,0),(-1,0),(0,1),(0,-1))
H, W = MAP()
grid = list2d(H, W, 0)
for i in range(INT()):
    a, b = MAP()
    grid[a-1][b-1] = 1

dp = list2d(H, W, 0)
dp[0][0] = 1
for i in range(H):
    for j in range(W):
        if not grid[i][j]:
            if i != 0:
                dp[i][j] += dp[i-1][j]
            if j != 0:
                dp[i][j] += dp[i][j-1]
print(dp[H-1][W-1])
