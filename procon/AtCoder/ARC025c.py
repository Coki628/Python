# -*- coding: utf-8 -*-

"""
・自力AC！
・各頂点からダイクストラして、各頂点から二分探索する。
・pypyAC5.6秒。(制約7秒)
"""

import sys
from bisect import bisect_right

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

N, M, R, T = MAP()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

RT = R / T
ans = 0
for i in range(N):
    # 頂点iをゴールとして、他の各頂点からの最短距離
    res = dijkstra(N, nodes, i)
    res.sort()
    for j in range(1, N):
        # ゴール以外の各頂点について、頂点jをカメの始点とした時、ウサギの始点候補がいくつあるか
        idx = bisect_right(res, res[j]*RT)
        cnt = N - idx
        # カメのが速い場合は、自分のいる頂点を含んでしまうので減らす
        if R < T:
            cnt -= 1
        ans += cnt
print(ans)
