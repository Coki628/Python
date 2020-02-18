# -*- coding: utf-8 -*-

"""
参考：https://rsk0315.hatenablog.com/entry/2019/12/29/051900
・自力ならず。。これは悔しい。
・グリッドダイクストラ
・3点からダイクストラして、全マスを経由候補として全探索。
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

def dijkstra(H: int, W: int, grid: list, src: tuple) -> list:
    """ グリッドダイクストラ(縦, 横, H*Wグリッド, 始点(h, w)) """
    from heapq import heappush, heappop

    # res[h][w] := マス(h, w)への最小コスト
    res = list2d(H, W, INF)
    # スタート位置
    srch, srcw = src
    que = [(0, srch, srcw)]
    res[srch][srcw] = 0
    directions = ((1,0),(-1,0),(0,1),(0,-1))
    # キューが空になるまで
    while que:
        # srcからの距離, マス(h, w)
        dist, curh, curw = heappop(que)
        if res[curh][curw] < dist:
            continue
        # マスcurからの4方向でループ
        for dh, dw in directions:
            nxth, nxtw = curh + dh, curw + dw
            cost = grid[nxth][nxtw]
             # 今回の経路のが短い時
            if dist + cost < res[nxth][nxtw]:
                res[nxth][nxtw] = dist + cost
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (dist+cost, nxth, nxtw))
    # マスsrcからの最短距離リストを返却
    return res

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

H, W = MAP()
grid = build_grid(H, W, INF, int, padding=1)

# 3点からダイクストラ
res1 = dijkstra(H+2, W+2, grid, (H, 1))
res2 = dijkstra(H+2, W+2, grid, (1, W))
res3 = dijkstra(H+2, W+2, grid, (H, W))

# マスを全探索
ans = INF
for h in range(1, H+1):
    for w in range(1, W+1):
        # 3点からの到達コスト - このマス自身は3回分入っているので2回分引く
        cost = res1[h][w] + res2[h][w] + res3[h][w] - grid[h][w] * 2
        ans = min(ans, cost)
print(ans)
