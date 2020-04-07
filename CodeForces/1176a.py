# -*- coding: utf-8 -*-

"""
・グラフに帰着、BFS
・各数値を頂点としたグラフで1を目指して操作する。
・でもみんなもっと普通にさくっと解いてた。。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bfs(src):
    from collections import deque, defaultdict

    que = deque([(src, 0)])
    dist = defaultdict(lambda: -1)
    dist[src] = 0
    while que:
        u, c = que.popleft()
        if u % 2 == 0:
            v = u // 2
            if dist[v] != -1:
                continue
            dist[v] = c + 1
            que.append((v, c+1))
        if u % 3 == 0:
            v = u // 3 * 2
            if dist[v] != -1:
                continue
            dist[v] = c + 1
            que.append((v, c+1))
        if u % 5 == 0:
            v = u // 5 * 4
            if dist[v] != -1:
                continue
            dist[v] = c + 1
            que.append((v, c+1))
    return dist

for _ in range(INT()):
    N = INT()
    res = bfs(N)
    print(res[1])
