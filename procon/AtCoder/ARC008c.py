# -*- coding: utf-8 -*-

"""
・自力TLE,MLE
・解法としては合ってた。
・初期AtCoderのメモリ制限64Mに苦しまされた。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
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

N = INT()
XYTR = []
for i in range(N):
    x, y, t, r = MAP()
    XYTR.append((x, y, t, r))

# 全ペアについてコスト計算して辺を張る
nodes = [[] for i in range(N)]
for i in range(N):
    x, y, t, r = XYTR[i]
    for j in range(i+1, N):
        x2, y2, t2, r2 = XYTR[j]
        dist = hypot(x-x2, y-y2) / min(t, r2)
        nodes[i].append((j, dist))
        dist = hypot(x-x2, y-y2) / min(t2, r)
        nodes[j].append((i, dist))

# 自分からの最短距離
res = dijkstra(N, nodes, 0)

res.sort(reverse=True)
ans = 0
for i in range(N-1):
    # 1秒1回しか投げれないので、後回しにする程コストとして秒数が乗る
    ans = max(ans, res[i] + i)
print(ans)
