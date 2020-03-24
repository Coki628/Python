# -*- coding: utf-8 -*-

"""
・自力AC！
・グラフ2回構築する感じ。
・ダイクストラ内で5000頂点あるのに毎回BFSするのは計算量心配だったけど、
　重みなしグラフで結局毎回O(M)だから、全頂点からやったとしてもO(NM)で済むんだな。
・とはいえpythonTLE,pypyAC4.0秒。(制約10秒)
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

def bfs(nodes, src):
    """ BFS(一般グラフ、重みなし) """
    from collections import deque

    N = len(nodes)
    que = deque([(src, 0)])
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
        c, r = CR[cur]
        # BFSでこの頂点から辺を張れる場所を調べる
        for nxt, d in enumerate(bfs(nodes, cur)):
            if d <= r and dist + c < res[nxt]:
                res[nxt] = dist + c
                heappush(que, (dist+c)*N+nxt)
    return res

N, M = MAP()
CR = []
for i in range(N):
    c, r = MAP()
    CR.append((c, r))
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

res = dijkstra(nodes, 0)
print(res[N-1])
