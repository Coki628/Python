# -*- coding: utf-8 -*-

"""
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def build_grid(H, W, intv, _type, space=True, padding=False):
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

H, W = MAP()
grid = build_grid(H, W, '', str, space=0)

cells = [[] for i in range(11)]
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            cells[0].append((i, j))
        elif grid[i][j] == 'G':
            cells[10].append((i, j))
        else:
            cells[int(grid[i][j])].append((i, j))

ans = 0
dp = list2d(H, W, INF)
h, w = cells[0][0]
dp[h][w] = 0
for i in range(10):
    for h1, w1 in cells[i]:
        for h2, w2 in cells[i+1]:
            dp[h2][w2] = min(dp[h2][w2] , dp[h1][w1] + abs(h1-h2) + abs(w1-w2))
h, w = cells[10][0]
ans = dp[h][w]
if ans == INF:
    print(-1)
else:
    print(ans)
