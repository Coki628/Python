# -*- coding: utf-8 -*-

"""
・類題攻略(nikkei2019-2-qual_dより)
・見た目は違うけど中身はほぼ同じ。
・辺を工夫してダイクストラ。
・でもこっちは問題の見た目的に最短経路問題に帰着させる発想が難しそう。
　公式解もセグ木だったし。
・pythonAC0.6秒、pypyAC0.5秒。
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
MOD = 998244353

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

M, N = MAP()
nodes = [[] for i in range(N+1)]
# 1つ手前の頂点に戻るコスト0の辺を張る
for i in range(N):
    nodes[i+1].append((i, 0))

for i in range(M):
    l, r, c = MAP()
    # 一番遠くまで行く辺を張る
    nodes[l].append((r, c))

res = dijkstra(N+1, nodes, 0)

if res[N] != INF:
    print(res[N])
else:
    print(-1)
