# -*- coding: utf-8 -*-

"""
・終了1時間後くらいでAC。(pythonで通ってれば30分後くらい)
・python,pypyは両方5000ループ内で5000再帰は間に合わずTLEで、C++で通した。
・と思ったけどhackされてて、紆余曲折あって最終的にはAC。詳細はC++のコメントで。
"""

import sys
from collections import defaultdict

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

D = defaultdict(int)
def rec(u, prev):
    global is_ok
    if u == X:
        a, b = min(u, prev), max(u, prev)
        if g >= D[(a, b)]:
            is_ok = True
            D[(a, b)] = g
        return True
    for v in nodes[u]:
        if v == prev:
            continue
        if rec(v, u):
            a, b = min(u, prev), max(u, prev)
            if a == -1:
                return True
            if g >= D[(a, b)]:
                is_ok = True
                D[(a, b)] = g
            return True
    return False

N = INT()
nodes = [[] for i in range(N)]
edges = {}
for i in range(N-1):
    a, b = MAP()
    a -= 1; b -= 1
    nodes[a].append(b)
    nodes[b].append(a)
    edges[(min(a, b), max(a, b))] = i

M = INT()
Qs = []
for i in range(M):
    a, X, g = MAP()
    a -= 1; X -= 1
    Qs.append((a, X, g))
for a, X, g in Qs:
    rec(a, -1)
for a, X, g in Qs:
    is_ok = False
    rec(a, -1)
    if not is_ok:
        print(-1)
        exit()

ans = [10**6] * (N-1)
for k, v in D.items():
    idx = edges[k]
    ans[idx] = v
print(*ans)
