# -*- coding: utf-8 -*-

"""
・蟻本演習2-5-2、自力AC！
・頂点拡張ダイクストラ
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

N, M = MAP()
# 4,7の倍数判定のために、28でmodした距離毎に頂点を用意する
nodes = [[] for i in range(N*28)]
for i in range(M):
    a, b, c = MAP()
    # N-1の街からは出られない
    if a == N-1:
        for i in range(28):
            nodes[b*28+i].append((a*28+(i+c)%28, c))
    elif b == N-1:
        for i in range(28):
            nodes[a*28+i].append((b*28+(i+c)%28, c))
    else:
        for i in range(28):
            nodes[a*28+i].append((b*28+(i+c)%28, c))
            nodes[b*28+i].append((a*28+(i+c)%28, c))

res = dijkstra(N*28, nodes, 0)

ans = INF
for i in [0,4,7,8,12,14,16,20,21,24]:
    # 4,7の倍数でN-1の街に到達した最小を取る
    ans = min(ans, res[(N-1)*28+i])
print(ans)
