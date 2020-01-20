# -*- coding: utf-8 -*-

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

def bfs(N, nodes, src):
    """ BFS(一般グラフ、重みなし) """
    from collections import deque

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

N = INT()
M = INT()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)

res = bfs(N, nodes, 0)

ans = 0
for i in range(1, N):
    # 最短距離が2(友達の友達)以内のものを数える
    if res[i] <= 2:
        ans += 1
print(ans)
