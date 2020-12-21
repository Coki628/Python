# -*- coding: utf-8 -*-

"""
・ABC164eで頂点拡張ダイクストラやって気づいたことがあったので。
・買い足す遷移は場所移動前に1回先だけやればいい。
・でもこっちは速くならなくて、普通にこれもTLE。。。
"""

import sys

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def dijkstra(nodes: list, src: int) -> list:
    from heapq import heappop, heappush

    N = len(nodes)
    res = list2d(N, K*2, INF)
    que = [(0, src, 0)]
    res[src][0] = 0
    while que:
        dist, u, f = heappop(que)
        if res[u][f] < dist:
            continue
        # 花を買い足す遷移
        if f < K and dist + Y[u] < res[u][f+X[u]]:
            res[u][f+X[u]] = dist + Y[u]
            heappush(que, (dist+Y[u], u, f+X[u]))
        # 各駅への遷移
        for v, cost in nodes[u]:
            if dist + cost < res[v][f]:
                res[v][f] = dist + cost
                heappush(que, (dist+cost, v, f))
    return res

N, M, K = MAP()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))
X = [0] * N
Y = [0] * N
for i in range(N):
    X[i], Y[i] = MAP()

res = dijkstra(nodes, 0)
ans = INF
# 花をK本以上買った頂点Nから最小値を取る
for k in range(K, K*2):
    ans = min(ans, res[N-1][k])
if ans != INF:
    print(ans)
else:
    print(-1)
