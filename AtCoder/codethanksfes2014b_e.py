# -*- coding: utf-8 -*-

"""
・2次元いもす、グリッドBFS
・実はN<=1000でグリッドが50*50=2500で最大1000*2500=250万だから
　いもすとかやらなくても愚直に塗って普通に間に合う設定だったっぽい。
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

def bfs(grid, src):
    """ BFS(グリッド、重みなし) """
    from collections import deque

    H, W = len(grid), len(grid[0])
    h, w = src
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    que = deque([(h, w, 0)])
    dist = list2d(H, W, INF)
    while que:
        h, w, c = que.popleft()
        if dist[h][w] != INF:
            continue
        if grid[h][w] == '#':
            continue
        dist[h][w] = c
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            que.append((h2, w2, c+1))
    return dist

H, W = MAP()
sh, sw = MAP()
gh, gw = MAP()
imos = list2d(H+2, W+2, 0)
N = INT()
for i in range(N):
    h1, w1, h2, w2 = MAP()
    h2 = h1 + h2
    w2 = w1 + w2
    # 2次元いもす
    imos[h1][w1] += 1
    imos[h1][w2] -= 1
    imos[h2][w1] -= 1
    imos[h2][w2] += 1
for i in range(H+2):
    for j in range(W+1):
        imos[i][j+1] += imos[i][j]
for j in range(W+2):
    for i in range(H+1):
        imos[i+1][j] += imos[i][j]

# 黒く塗られた場所だけ通れるようにする
grid = list2d(H+2, W+2, '#')
for i in range(1, H+1):
    for j in range(1, W+1):
        if imos[i][j]:
            grid[i][j] = '.'

# グリッドBFS
res = bfs(grid, (sh, sw))
if res[gh][gw] != INF:
    YES()
else:
    NO()
