# -*- coding: utf-8 -*-

"""
・木の直径(木の最遠節点間の距離)
・経路探索2回で出せる。
"""

import sys
from heapq import heappop, heappush

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

# ダイクストラ(頂点数, 隣接リスト(0-indexed), 始点)
def dijkstra(N: int, nodes: list, src: int) -> list:
    # 頂点[ある始点からの最短距離]
    # (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [float('inf')] * N
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
            if res[cur] + cost < res[nxt]:
                res[nxt] = res[cur] + cost
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (res[nxt], nxt))
    # ノードsrcからの最短距離リストを返却
    return res

N=INT()
nodes=[[] for i in range(N)]
for i in range(N-1):
    u,v,w=MAP()
    nodes[u].append((v, w))
    nodes[v].append((u, w))

res=dijkstra(N, nodes, 0)
x=res.index(max(res))
res=dijkstra(N, nodes, x)
print(max(res))
