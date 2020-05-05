# -*- coding: utf-8 -*-

"""
・約数列挙貼って一撃、と思ったら違った。N=18まであるので、TLEする。
・グラフに帰着のBFS。10冪なので、10の約数で割っていく。
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

def bfs(src):
    from collections import deque

    que = deque([(src)])
    visited = set()
    visited.add(src)
    while que:
        u = que.popleft()
        for x in [2, 5]:
            if u % x:
                continue
            v = u // x
            if v in visited:
                continue
            visited.add(v)
            que.append((v))
    return visited

N = INT()
res = bfs(10**N)
for a in sorted(res):
    print(a)
