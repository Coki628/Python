# -*- coding: utf-8 -*-

"""
・グリッドBFS
・始点、終点両方から最短経路を取るとうまくいくやつ。
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
    global s, g
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
            if grid[i][j] == 'S':
                s = (i, j)
            if grid[i][j] == 'G':
                g = (i, j)
    return grid

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
        # 壁への到達距離を記録してからスキップ
        if grid[h][w] == '#':
            dist[h][w] = c
            continue
        dist[h][w] = c
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            que.append((h2, w2, c+1))
    return dist

H, W = MAP()
s = g = ()
grid = build_grid(H, W, '#', str, space=0, padding=1)

# 始点、終点から最短経路
res1 = bfs(grid, s)
res2 = bfs(grid, g)

ans = -1
for h in range(1, H+1):
    for w in range(1, W+1):
        # 両方から到達できる壁で、最も遠い場所を見つける
        if res1[h][w] != INF and res2[h][w] != INF:
            ans = max(ans, res1[h][w] + res2[h][w])
print(ans)
