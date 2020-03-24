# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・最短経路、グラフ2回構築
・10万頂点、20万辺で、pythonAC1.2秒、pypyAC0.8秒。
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
MOD = 10 ** 9 + 7

def bfs(nodes, src):
    """ BFS(一般グラフ、重みなし) """
    from collections import deque

    N = len(nodes)
    que = deque()
    # 多点スタート
    for u in src:
        que.append((u, 0))
    dist = [INF] * N
    while que:
        u, c = que.popleft()
        if dist[u] != INF:
            continue
        dist[u] = c
        for v in nodes[u]:
            que.append((v, c+1))
    return dist

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

N, M, K, S = MAP()
P, Q = MAP()
zombied = [k-1 for k in LIST(K)]
nodes = [[] for i in range(N)]
edges = []
for _ in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)
    edges.append((a, b))

res = bfs(nodes, zombied)
safe = [1] * N
for i, dist in enumerate(res):
    if dist == 0:
        safe[i] = -1
    elif dist <= S:
        safe[i] = 0

cost = [0] * N
for i in range(1, N-1):
    if safe[i] < 0:
        cost[i] = INF
    elif safe[i] == 0:
        cost[i] = Q
    else:
        cost[i] = P

nodes2 = [[] for i in range(N)]
for a, b in edges:
    nodes2[a].append((b, cost[b]))
    nodes2[b].append((a, cost[a]))

res = dijkstra(nodes2, 0)
ans = res[N-1]
print(ans)
