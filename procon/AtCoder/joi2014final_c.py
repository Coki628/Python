# -*- coding: utf-8 -*-

"""
・自力AC！
・最初ちょっと方針間違ったけど(三分探索して死亡)、ちゃんとすぐに修正できた。
・ダイクストラはやるだけ。その後の各距離でのコスト計算は、小さい方から順番に進めれば、
　最終的に線形時間で全部チェックできる。
・pypyでも1秒は間に合わず。C++でAC0.3秒。
"""

import sys
from collections import defaultdict

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
MOD = 10 ** 9 + 7

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ高速化版(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    res = [INF] * N
    que = [src]
    res[src] = 0
    while que:
        cur = heappop(que)
        dist = cur // N
        cur %= N
        if res[cur] < dist:
            continue
        for nxt, cost in nodes[cur]:
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                heappush(que, (dist+cost)*N+nxt)
    return res

N, M, C = MAP()
nodes = [[] for i in range(N)]
nodes2 = [[] for i in range(N)]
dsm = 0
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))
    # 各頂点、どの辺が繋がっているかも覚えておく
    nodes2[a].append((c, i))
    nodes2[b].append((c, i))
    dsm += c

dist = dijkstra(nodes, 0)
D = defaultdict(list)
for i, d in enumerate(dist):
    # 頂点0からの距離毎にまとめておく
    D[d].append(i)
D = sorted(D.items())
edgecnt = [2] * M
ans = INF
# 距離が近い方から順番に見ていく
for x, li in D:
    for u in li:
        for c, i in nodes2[u]:
            edgecnt[i] -= 1
            # 2回出現したら、この辺はもう不要
            if edgecnt[i] == 0:
                dsm -= c
    # この距離の頂点と繋がる辺を全てチェックしたら、コストを計算
    ans = min(ans, C * x + dsm)
print(ans)
