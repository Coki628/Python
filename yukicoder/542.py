# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・グラフに帰着、BFS。
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

def bfs(a, b):
    from collections import deque

    que = deque([(0, a, b)])
    visited = set()
    while que:
        u, a, b = que.popleft()
        if a > 0:
            v = u + 1
            if v not in visited:
                visited.add(v)
                que.append((v, a-1, b))
        if b > 0:
            v = u + 5
            if v not in visited:
                visited.add(v)
                que.append((v, a, b-1))
    return visited

a, b = MAP()

res = bfs(a, b)
for a in sorted(res):
    print(a)
