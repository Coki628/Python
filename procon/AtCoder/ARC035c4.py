# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc035
・色々悩んだけど自力ならず。。
・辺が1本変わった時に更新する必要があるのは(コード内コメント参照)、、言われてみればそうだねって感じ。。
・400頂点のワーシャルフロイド(400^3=6400万)がpypyでも通らずTLE。。
・辺が最大でも1000〜1400にしかならないので、ダイクストラを全頂点にやった方が速い。
・これでpypyAC1.7秒。
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

N, M = MAP()
G = list2d(N, N, INF)
nodes = [[] for i in range(N)]
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    G[a][b] = c
    G[b][a] = c
    nodes[a].append((b, c))
    nodes[b].append((a, c))

# 辺の数が少ないので、ワーシャルフロイドの代わりに全頂点からダイクストラ
res = [None] * N
for i in range(N):
    res[i] = dijkstra(N, nodes, i)

# 頂点a,b間の辺が変更された時の最短路更新
def update(G, a, b):
    for i in range(N):
        for j in range(N):
            G[i][j] = min(
                G[i][j],
                # a ~ b を通るルート
                G[i][a] + c + G[b][j],
                # b ~ a を通るルート
                G[i][b] + c + G[a][j],
            )
    return G

sm = sum([sum(row) for row in res])
for _ in range(INT()):
    a, b, c = MAP()
    a -= 1; b -= 1
    if c < G[a][b]:
        G[a][b] = c
        G[b][a] = c
        res = update(res, a, b)
        sm = sum([sum(row) for row in res])
    print(sm // 2)
