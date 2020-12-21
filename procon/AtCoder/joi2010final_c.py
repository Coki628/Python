# -*- coding: utf-8 -*-

"""
・自力AC！
・最短経路、ダイクストラ
・多点スタートのダイクストラ初めて使ったけど、ちゃんと動いて良かった。
　BFSでやったことあるのと同じ要領で、多分できるだろうとは思ったけど。
・制約0.5秒64MBはpythonもpypyもきつい。。TLE,MLEにて、C++でAC0.04秒。
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

def dijkstra(nodes: list, src: list) -> list:
    from heapq import heappush, heappop

    N = len(nodes)
    res = [INF] * N
    que = []
    # 多点スタート
    for u in src:
        que.append(u)
        res[u] = 0
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

N, M, K = MAP()
nodes = [[] for i in range(N)]
for _ in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))
src = [a-1 for a in LIST(K)]

# モールのある全ての町を始点にダイクストラ
res = dijkstra(nodes, src)
ans = 0
for u in range(N):
    for v, c in nodes[u]:
        # uからモールへの距離 + この辺の長さ + vからモールへの距離 の真ん中
        ans = max(ans, ceil(res[u] + c + res[v], 2))
print(ans)
