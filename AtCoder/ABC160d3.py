# -*- coding: utf-8 -*-

"""
・BFSライブラリ変更のため動作確認
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
    dist[src] = 0
    while que:
        u, c = que.popleft()
        for v in nodes[u]:
            if dist[v] != INF:
                continue
            dist[v] = c + 1
            que.append((v, c+1))
    return dist

N, x, y = MAP()
x -= 1; y -= 1

nodes = [[] for i in range(N)]
for i in range(N-1):
    nodes[i].append(i+1)
    nodes[i+1].append(i)
nodes[x].append(y)
nodes[y].append(x)

distx = bfs(nodes, x)
disty = bfs(nodes, y)

G = list2d(N, N, 0)
for i in range(N):
    for j in range(i+1, N):
        G[i][j] = min(j - i, distx[i] + disty[j] + 1, distx[j] + disty[i] + 1)
ans = [0] * N
for i in range(N):
    for j in range(i+1, N):
        ans[G[i][j]] += 1
[print(a) for a in ans[1:]]
