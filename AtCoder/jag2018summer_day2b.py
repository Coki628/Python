# -*- coding: utf-8 -*-

"""
・自力AC
・グラフに帰着、BFS
・各金額についてのコインの最小使用回数はBFSすると分かる。
・実験してみると、14以降くらいからずっと500なのでそこまでやる。
"""

import sys
from collections import Counter

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

def bfs():
    from collections import deque

    coins = (1, 5, 10, 50, 100, 500)
    que = deque()
    dist = {}
    for coin in coins:
        que.append((coin, 1))
        dist[coin] = 1
    while que:
        u, c = que.popleft()
        for coin in coins:
            v = u + coin
            if v in dist or c >= N:
                continue
            dist[v] = c + 1
            que.append((v, c+1))
    return dist

N = INT()

N = min(N, 14)
res = bfs()
ans = Counter(res.values())
print(ans[N])
