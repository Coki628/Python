# -*- coding: utf-8 -*-

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

# 右左上下
directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
# 斜め有り
directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, -1), (1, 1), (-1, 1))
# 四方に一回り大きいグリッドを作る
# grid = list2d(H+2, W+2, '*')
# for i in range(1, H+1):
#     row = list(input())
#     for j in range(1, W+1):
#         grid[i][j] = row[j-1]

def build_grid(H, W, intv, _type, space=False, padding=True):
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

def rot90(grid):
    """ グリッドを時計回りに90度回転 """
    H = len(grid)
    W = len(grid[0])
    res = list2d(W, H, 0)
    for i in range(H):
        for j in range(W):
            res[j][H-i-1] = grid[i][j]
    return res

def scale(grid, K):
    """ グリッドをK倍に拡大 """
    H, W = len(grid), len(grid[0])
    res = list2d(H*K, W*K, '')
    for i in range(H*K):
        for j in range(W*K):
            res[i][j] = grid[i//K][j//K]
    return res

# ※多点スタート対応のためsrcにはlistを渡す
def bfs(grid, src):
    """ BFS(グリッド、重みなし) """
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dist = list2d(H, W, INF)
    que = deque()
    for h, w in src:
        que.append((h, w))
        dist[h][w] = 0
    while que:
        h, w = que.popleft()
        for dh, dw in directions:
            nh = h + dh
            nw = w + dw
            if grid[nh][nw] == -1:
                continue
            if dist[nh][nw] != INF:
                continue
            dist[nh][nw] = dist[h][w] + 1
            que.append((nh, nw))
    return dist

def dijkstra(grid: list, src: tuple) -> list:
    """ グリッドダイクストラ(縦, 横, H*Wグリッド, 始点(h, w)) """
    from heapq import heappush, heappop

    H, W = len(grid), len(grid[0])
    # res[h][w] := マス(h, w)への最小コスト
    res = list2d(H, W, INF)
    # スタート位置
    srch, srcw = src
    que = [(0, srch, srcw)]
    res[srch][srcw] = 0
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
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
