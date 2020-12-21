# -*- coding: utf-8 -*-

"""
・numpy実装
・2次元まとめて一括処理できるのはありがたい。
"""

import sys
import numpy as np

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

INF = 10 ** 18
H, W = MAP()
grid = [None] * H
for i in range(H):
    grid[i] = LIST()
grid = np.array(grid, dtype=np.int64)

def check(h, w):
    dist = np.full((H, W), INF, dtype=np.int64)
    # 横方向の距離
    for i in range(H):
        cnt = 0
        for j in range(w, W):
            dist[i,j] = min(dist[i,j], cnt)
            cnt += 1
        cnt = 1
        for j in range(w-1, -1, -1):
            dist[i,j] = min(dist[i,j], cnt)
            cnt += 1
    # 縦方向の距離
    for j in range(W):
        cnt = 0
        for i in range(h, H):
            dist[i,j] = min(dist[i,j], cnt)
            cnt += 1
        cnt = 1
        for i in range(h-1, -1, -1):
            dist[i,j] = min(dist[i,j], cnt)
            cnt += 1
    # 2次元全部まとめて距離*人数
    cost = dist * grid
    return cost.sum()

ans = INF
for i in range(H):
    for j in range(W):
        ans = min(ans, check(i, j))
print(ans)
