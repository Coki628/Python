# -*- coding: utf-8 -*-

"""
・グリッドダイクストラ
・六角形の区画は、限定的に斜めの辺も張ったグリッドだと思えばいい。
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

def dijkstra(H: int, W: int, grid: list, src: tuple) -> list:
    """ グリッドダイクストラ(縦, 横, H*Wグリッド, 始点(h, w)) """
    from heapq import heappush, heappop

    # res[h][w] := マス(h, w)への最小コスト
    res = list2d(H, W, INF)
    srch, srcw = src
    que = [(0, srch, srcw)]
    res[srch][srcw] = 0
    # 斜めの方向を追加
    directions = ((1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,-1),(1,1),(-1,1))
    while que:
        dist, curh, curw = heappop(que)
        if res[curh][curw] < dist:
            continue
        for i, (dh, dw) in enumerate(directions):
            # 偶数行は左上、左下は行かない
            if curh % 2 == 0 and i in [4, 5]:
                continue
            # 奇数行は右上、右下は行かない
            if curh % 2 == 1 and i in [6, 7]:
                continue
            nxth, nxtw = curh + dh, curw + dw
            cost = grid[nxth][nxtw]
            if dist + cost < res[nxth][nxtw]:
                res[nxth][nxtw] = dist + cost
                heappush(que, (dist+cost, nxth, nxtw))
    return res

H, W = MAP()
grid = build_grid(H, W, '#', str, space=0, padding=1)
s = t = ()
for i in range(H+2):
    for j in range(W+2):
        if grid[i][j] == '#':
            grid[i][j] = INF
        elif grid[i][j] == 's':
            s = (i, j)
            grid[i][j] = 0
        elif grid[i][j] == 't':
            t = (i, j)
            grid[i][j] = 0
        else:
            grid[i][j] = int(grid[i][j])

res = dijkstra(H+2, W+2, grid, s)
print(res[t[0]][t[1]])
