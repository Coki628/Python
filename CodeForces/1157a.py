# -*- coding: utf-8 -*-

"""
・グラフに帰着。BFS。整数を頂点に。
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
    from collections import deque

    que = deque([src])
    dist = set()
    dist.add(src)
    while que:
        u = que.popleft()
        v  = u + 1
        while v % 10 == 0:
            v //= 10
        if v in dist:
            continue
        dist.add(v)
        que.append(v)
    return dist

N = INT()

res = bfs(N)
ans = len(res)
print(ans)
