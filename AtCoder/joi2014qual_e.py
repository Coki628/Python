# -*- coding: utf-8 -*-

"""
・自力AC！
・グリッドBFS応用、8方向、多点スタート
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, -1), (1, 1), (-1, 1))
    que = deque()
    visited = list2d(H, W, 0)
    for h, w in src:
        que.append((h, w, 1))
        visited[h][w] = 1
    res = 0
    while que:
        h, w, c = que.popleft()
        res = max(res, c)
        for dh, dw in directions:
            nh = h + dh
            nw = w + dw
            if visited[nh][nw]:
                continue
            # 周囲8方向について、ここが崩れる分の強度を引く
            grid[nh][nw] -= 1
            # 強度が足りなくなる所があったら、次回以降崩すキューに追加
            if grid[nh][nw] <= 0:
                visited[nh][nw] = 1
                que.append((nh, nw, c+1))
    return res

H, W = MAP()
grid = build_grid(H, W, '', str, space=0)

directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1,-1), (-1, -1), (1, 1), (-1, 1))
cnt = list2d(H, W, INF)
for h in range(1, H-1):
    for w in range(1, W-1):
        if grid[h][w] != '.':
            cnt[h][w] = int(grid[h][w])
            for dh, dw in directions:
                nh, nw = h + dh, w + dw
                if grid[nh][nw] == '.':
                    cnt[h][w] -= 1
        else:
            cnt[h][w] = INF

src = []
for h in range(1, H-1):
    for w in range(1, W-1):
        if cnt[h][w] <= 0:
            # 初回の波で崩れるマス
            src.append((h, w))
# 多点スタートでBFS
res = bfs(cnt, src)
print(res)
