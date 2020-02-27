# -*- coding: utf-8 -*-

"""
・400点自力AC
・グリッドBFS
・制約が小さい(最大400マス)ので、全マスから最短距離調べてOK。
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
    dist = list2d(H+2, W+2, -1)
    while que:
        h, w, c = que.popleft()
        if dist[h][w] != -1:
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
grid = build_grid(H, W, '#', str, space=0, padding=1)

ans = 0
for i in range(1, H+1):
    for j in range(1, W+1):
        if grid[i][j] != '#':
            # 有効な全マスから最短距離試す
            res = bfs(grid, (i, j))
            for k in range(1, H+1):
                for l in range(1, W+1):
                    # 一番遠い場所
                    ans = max(ans, res[k][l])
print(ans)
