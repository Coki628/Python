# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day1/editorial-I.pdf
・公式解に合わせてダイクストラもやってみる。
"""

import sys
from heapq import heappush, heappop

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

# ダイクストラ(頂点数, 隣接リスト(0-indexed), 始点, 開始グループ)
def dijkstra(N: int, nodes: list, src: int, startgroup:int) -> list:
    # 頂点[ある始点からの最短距離]
    # (経路自体を知りたい時はここに前の頂点も持たせる)
    res = [INF] * N
    # スタート位置
    que = [(0, src, startgroup)]
    res[src] = 0
    # キューが空になるまで
    while len(que) != 0:
        # srcからの距離, 現在のノード, 現在のグループ
        dist, cur, curgroup = heappop(que)
        # 出発ノードcurの到着ノードでループ
        for nxt, nxtgroup in nodes[cur]:
            # グループが一致していればコスト0
            if curgroup==nxtgroup:
                cost=0
            else:
                cost=1
            # 今回の経路のが短い時
            if res[cur] + cost < res[nxt]:
                res[nxt] = res[cur] + cost
                # 現在の移動距離をキューの優先度として、早い方から先に処理するようにする
                heappush(que, (res[nxt], nxt, nxtgroup))
    # ノードsrcからの最短距離リストを返却
    return res

N,M,K=MAP()
nodes=[[] for i in range(N)]
for i in range(M):
    a,b,group=MAP()
    nodes[a-1].append((b-1, group))
    nodes[b-1].append((a-1, group))

ans=INF
for nxt, group in nodes[0]:
    # 最初の移動はどれもコスト0なので全通り試す
    res=dijkstra(N, nodes, nxt, group)
    ans=min(ans, res[N-1])
if ans!=INF:
    print((ans+1)*K)
else:
    print(-1)
