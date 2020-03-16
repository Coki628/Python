# -*- coding: utf-8 -*-

"""
・グリッドBFS、最短経路
・経由地を起点にすると一発で出るやつ。
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
    return grid

def bfs(grid, src):
    """ BFS(グリッド、重みなし) """
    from collections import deque

    H, W = len(grid), len(grid[0])
    h, w = src
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    que = deque([(h, w, 0)])
    dist = list2d(H+2, W+2, INF)
    while que:
        h, w, c = que.popleft()
        if dist[h][w] != INF:
            continue
        dist[h][w] = c
        for dh, dw in directions:
            h2 = h + dh
            w2 = w + dw
            if grid[h2][w2] == '#':
                continue
            que.append((h2, w2, c+1))
    return dist

H, W = MAP()
grid = build_grid(H, W, '', str, space=0)

for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            si, sj = i, j
        elif grid[i][j] == 'G':
            gi, gj = i, j
        elif grid[i][j] == 'C':
            src = (i, j)

res = bfs(grid, src)

if res[si][sj] == INF or res[gi][gj] == INF:
    print(-1)
    exit()

ans = res[si][sj] + res[gi][gj]
print(ans)
