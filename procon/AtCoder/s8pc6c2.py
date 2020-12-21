# -*- coding: utf-8 -*-

"""
・400点自力AC
・グリッドで普通のBFS
・横移動チェック入れたらACした。
"""

import sys
import numpy as np
from collections import deque

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

H, W = MAP()

grid_org = np.full((H, W), '', dtype=np.str)
for i in range(H):
    grid_org[i] = np.array(list(input()), dtype=np.str)

# 横移動がずっとできる場所があるかどうか
is_ok = False
for i in range(H):
    if np.count_nonzero(grid_org[i]=='.') == W:
        is_ok = True

def bfs(grid, sh, sw):
    res = [[-1]*(W+2) for i in range(H+2)]
    # 今回2方向しか行けない
    directions = [(0,1), (1,0)]
    que = deque()
    que.append((sh, sw, 0))
    while len(que):
        h1, w1, cost = que.popleft()
        if res[h1][w1] != -1:
            continue
        res[h1][w1] = cost
        for h, w in directions:
            h2 = h1 + h
            w2 = w1 + w
            if grid[h2][w2] == '*' or grid[h2][w2] == '#':
                continue
            if grid[h2][w2] == '.':
                que.append((h2, w2, cost+1))
    return res

grid = np.full((H+2, W+2), '*', dtype=np.str)
grid[1:-1,1:-1] = grid_org
res = bfs(grid, 1, 1)

if is_ok and res[H][W] != -1:
    print('Yay!')
else:
    print(':(')
