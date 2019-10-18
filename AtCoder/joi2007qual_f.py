# -*- coding: utf-8 -*-

"""
・蟻本演習2-5-2、さくっと自力AC
・ダイクストラ
・辺を動的更新しながら何回もダイクストラするのはやってないパターンだったから
　計算量心配だったけど、頂点100クエリ5000で、pythonAC1.9秒、pypyAC1.0秒。
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

N, K = MAP()
nodes = [[] for i in range(N)]
for i in range(K):
    li = LIST()
    if li[0] == 0:
        a, b = li[1:]
        a -= 1
        b -= 1
        res = dijkstra(N, nodes, a)
        if res[b] == INF:
            print(-1)
        else:
            print(res[b])
    else:
        c, d, e = li[1:]
        c -= 1
        d -= 1
        nodes[c].append((d, e))
        nodes[d].append((c, e))
