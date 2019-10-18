# -*- coding: utf-8 -*-

"""
・蟻本演習2-5-2、自力AC！嬉しいけど、解法すぐ出なくて危なかった。。
・ダイクストラ
"""

import sys
from itertools import accumulate

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
INF = float('inf')
MOD = 10 ** 9 + 7

def dijkstra(N: int, nodes: list, src: int) -> list:
    """ ダイクストラ高速化版(頂点数, 隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    # 頂点[ある始点からの最短距離]
    res = [INF] * N
    # スタート位置
    que = [src]
    res[src] = 0
    # キューが空になるまで
    while len(que) != 0:
        # 距離*N + 現在のノード
        cur = heappop(que)
        # 距離とノードに分ける
        dist = cur // N
        cur %= N
        # 出発ノードcurの到着ノードでループ
        for nxt, cost in nodes[cur]:
            # 今回の経路のが短い時
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                # 距離*N+ノード番号 の形でキューに詰める
                heappush(que, (dist+cost)*N+nxt)
    # ノードsrcからの最短距離リストを返却
    return res

N, M, s, t = MAP()
s -= 1
t -= 1
nodes1 = tuple([] for i in range(N))
nodes2 = tuple([] for i in range(N))
for i in range(M):
    u, v, a, b = MAP()
    u -= 1
    v -= 1
    nodes1[u].append((v, a))
    nodes1[v].append((u, a))
    nodes2[u].append((v, b))
    nodes2[v].append((u, b))

res1 = dijkstra(N, nodes1, s)
res2 = dijkstra(N, nodes2, t)

A = [0] * N
for i in range(N):
    # sからiまで円、iからtまでスヌークで行く
    A[i] = res1[i] + res2[i]

# 最後のが使える両替所が少ないので、後ろから累積minする
ans = list(accumulate(A[::-1], min))[::-1]
total = 10 ** 15
[print(total-a) for a in ans]
