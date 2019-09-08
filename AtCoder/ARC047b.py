# -*- coding: utf-8 -*-

"""
・座標変換、頑張ったけどWA。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

def calc(m):
    L = D = -INF
    R = U = INF
    for x, y in XY:
        u = x + y
        v = x - y
        l = u - m
        r = u + m
        d = v - m
        _u = v + m
        L = max(L, l)
        R = min(R, r)
        D = max(D, d)
        U = min(U, _u)
    return L <= R and D <= U

def bisearch_min(mn, mx, func):
    """ 条件を満たす最小値を見つける二分探索 """
    ok = mx
    ng = mn
    while ng+1 < ok:
        mid = (ok+ng) // 2
        if func(mid):
            # 下を探しに行く
            ok = mid
        else:
            # 上を探しに行く
            ng = mid
    return ok

res = bisearch_min(-1, 4*10**9, calc)

L = D = -INF
R = U = INF
for x, y in XY:
    u = x + y
    v = x - y
    l = u - res
    r = u + res
    d = v - res
    _u = v + res
    L = max(L, l)
    R = min(R, r)
    D = max(D, d)
    U = min(U, _u)

for u in range(L, R+1):
    for v in range(D, U+1):
        if (u+v)%2 == 0: 
            x = (u+v) // 2
            y = u - x
            print(x, y)
            exit()
