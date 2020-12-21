# -*- coding: utf-8 -*-

"""
・ちょっと悩んだけど自力AC！
・グリッドダイクストラ
・状態(添字)を1つ増やす。コストとは別に距離を持つ。
・そのままやっててTLEしたんだけど、キューに入れる優先度で
　コストより距離を先にしたら通った。
・計算量は30^4=81万にダイクストラのlogが乗る。
　pythonAC1.6秒、pypyAC1.1秒。
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

def dijkstra(grid: list, src: tuple) -> list:
    """ グリッドダイクストラ(縦, 横, H*Wグリッド, 始点(h, w)) """
    from heapq import heappush, heappop

    H, W = len(grid), len(grid[0])
    # res[i][j][k] := (i, j)マスにいて、(1, 1)からの移動距離がkの時の最小コスト
    res = list3d(H, W, H*W, INF)
    srch, srcw = src
    que = [(0, 0, srch, srcw)]
    res[srch][srcw][0] = 0
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while que:
        d2, dist, curh, curw = heappop(que)
        if res[curh][curw][d2] < dist:
            continue
        for dh, dw in directions:
            nxth, nxtw = curh + dh, curw + dw
            if grid[nxth][nxtw] == INF:
                continue
            cost = (d2*2+1) * grid[nxth][nxtw]
            if d2+1 < H*W and dist + cost < res[nxth][nxtw][d2+1]:
                res[nxth][nxtw][d2+1] = dist + cost
                heappush(que, (d2+1, dist+cost, nxth, nxtw))
    return res

H, W = MAP()
grid = build_grid(H, W, INF, int, padding=1)

res = dijkstra(grid, (1, 1))
ans = min(res[H][W])
print(ans)
