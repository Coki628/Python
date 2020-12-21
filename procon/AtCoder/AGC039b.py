# -*- coding: utf-8 -*-

"""
・これは参戦中にサンプルWA
・デバッグした結果、近い頂点から見なきゃダメだと思ってこの後DFSをBFSで書き直した。
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

N = INT()
G = [None] * N
for i in range(N):
    G[i] = list(map(int, input()))

nodes = [[] for i in range(N)]
for i in range(N):
    for j in range(N):
        if G[i][j]:
            nodes[i].append(j)
            nodes[j].append(i)

visited = list2d(N, N, 0)
group = [-1] * N
cnt = 0
def rec(u, cnt):
    if group[u] == -1:
        group[u] = cnt
    elif group[u] == cnt:
        return cnt
    else:
        return -1
    mx = cnt
    res = 0
    for v in nodes[u]:
        if visited[u][v]:
            continue
        visited[u][v] = visited[v][u] = 1
        res = rec(v, cnt+1)
        if res == -1:
            return -1
        else:
            mx = max(mx, res)
    return mx

ans = -1
for i in range(N):
    visited = list2d(N, N, 0)
    group = [-1] * N
    ans = max(ans, rec(i, 1))
print(ans)
