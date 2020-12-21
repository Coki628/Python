# -*- coding: utf-8 -*-

"""
・600点自力AC！(これはTLEでC++でAC)
・幾何と見せかけてグラフに帰着。
・各円を頂点とみなして、全円間で距離取って辺としてつなぐ。
"""

import sys
from math import hypot

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

def dijkstra(N: int, nodes: list, src: int) -> list:
    """ ダイクストラ(頂点数, 隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    # 頂点[ある始点からの最短距離] (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    # スタート位置
    que = [(0, src)]
    res[src] = 0
    # キューが空になるまで
    while len(que) != 0:
        # srcからの距離, 現在のノード
        dist, cur = heappop(que)
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (dist+cost, nxt))
    # ノードsrcからの最短距離リストを返却
    return res

sx, sy, tx, ty = MAP()
N = INT()
XYR = []
for i in range(N):
    x, y, r = MAP()
    XYR.append((x, y, r))
# 始点と終点も半径0の円とする
XYR.append((sx, sy, 0))
XYR.append((tx, ty, 0))
N += 2

# 各円を頂点としたグラフとみなす
nodes = [[] for i in range(N)]
for i in range(N):
    x1, y1, r1 = XYR[i]
    for j in range(i+1, N):
        x2, y2, r2 = XYR[j]
        dist = hypot(x1-x2, y1-y2)
        # 2円間の距離(重なっていれば0)
        cost = max(dist - (r1+r2), 0)
        nodes[i].append((j, cost))
        nodes[j].append((i, cost))

res = dijkstra(N, nodes, N-2)
print(res[N-1])
