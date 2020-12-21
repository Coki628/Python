# -*- coding: utf-8 -*-

"""
・さくっと自力AC
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

R, C, D = MAP()
grid = [None] * R
for i in range(R):
    grid[i] = LIST()

ans = 0
for i in range(R):
    for j in range(C):
        # 偶奇が大丈夫で距離も届くなら答え候補
        if (i+j)%2 == D%2 and (i+j) <= D:
            ans = max(ans, grid[i][j])
print(ans)
