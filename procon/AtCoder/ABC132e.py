# -*- coding: utf-8 -*-

"""
・参戦中。これはTLE
"""

import sys
from heapq import heappop, heappush

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

# ダイクストラ(頂点数, 隣接リスト(0-indexed), 始点)
def dijkstra(N: int, nodes: list, src: int) -> list:
    # 頂点[ある始点からの最短距離]
    # (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    # スタート位置
    que = [(0, src)]
    res[src] = 0
    # キューが空になるまで
    while len(que) != 0:
        # srcからの距離, 現在のノード
        dist, cur = heappop(que)
        # 出発ノードcurの到着ノードでループ
        for nxt in nodes[cur]:
            # 今回の経路のが短い時
            if res[cur] + 1 < res[nxt]:
                res[nxt] = res[cur] + 1
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (res[nxt], nxt))
    # ノードsrcからの最短距離リストを返却
    return res

N,M=MAP()
nodes=[[] for i in range(N)]
for i in range(M):
    u,v=MAP()
    nodes[u-1].append(v-1)
S,T=MAP()
S-=1
T-=1

edges2=[]
def rec(org, u, cnt):
    if visited[u][cnt]:
        return
    visited[u][cnt]=True
    if cnt==3:
        edges2.append((org, u))
        return
    for v in nodes[u]:
        rec(org, v, cnt+1)
    return

for i in range(N):
    visited=list2d(N, 4, False)
    rec(i, i, 0)

nodes2=[[] for i in range(N)]
for u, v in edges2:
    nodes2[u].append(v)

res=dijkstra(N, nodes2, S)
if res[T]==INF:
    print(-1)
else:
    print(res[T])
