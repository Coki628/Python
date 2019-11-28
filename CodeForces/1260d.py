# -*- coding: utf-8 -*-

"""
・自力WA
"""

import sys
from operator import itemgetter
from bisect import bisect_left

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

def bisearch_min(mn, mx, func):

    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

M, N, K, T = MAP()
A = LIST()

LRD = []
for i in range(K):
    l, r, d = MAP()
    LRD.append((l, r, d))
LRD.sort(key=itemgetter(2), reverse=True)
dmx = LRD[0][2]

def check(m):
    nxt = i = 0
    add = INF
    while i < K and LRD[i][2] >= m:
        l, r, _ = LRD[i]
        if r >= l:
            nxt = max(nxt, r)
        add = min(add, l)
        i += 1
    if add > nxt:
        add = nxt + 1
    return (nxt-(add-1))*2 + N+1 <= T

res = bisearch_min(0, dmx+1, check)
A.sort()
idx = bisect_left(A, res-1)
cnt = M - idx
print(cnt)
