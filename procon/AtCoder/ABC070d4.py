# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
・木の最短経路、DFS
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

def dfs(N, nodes, src):
    """ DFS(木、スタック、重みあり) """

    stack = [(src, -1, 0)]
    dist = [INF] * N
    while stack:
        u, prev, c1 = stack.pop()
        dist[u] = c1
        for v, c2 in nodes[u]:
            if v != prev:
                stack.append((v, u, c1+c2))
    return dist

N = INT()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b, c = MAP()
    a -= 1; b -= 1
    nodes[a].append((b, c))
    nodes[b].append((a, c))

Q, K = MAP()
K -= 1
res = dfs(N, nodes, K)
ans = []
for i in range(Q):
    x, y = MAP()
    x -= 1; y -= 1
    ans.append(str(res[x] + res[y]))
print('\n'.join(ans))
