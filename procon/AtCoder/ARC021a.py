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

H = W = 4

grid = list2d(H+2, W+2, 0)
for i in range(1, H+1):
    row = LIST()
    for j in range(1, W+1):
        grid[i][j] = row[j-1]

is_ok = False
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] == grid[i-1][j] \
                or grid[i][j] == grid[i+1][j] \
                or grid[i][j] == grid[i][j-1] \
                or grid[i][j] == grid[i][j+1]:
            is_ok = True
if is_ok:
    print('CONTINUE')
else:
    print('GAMEOVER')
