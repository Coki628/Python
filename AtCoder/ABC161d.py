# -*- coding: utf-8 -*-

"""
・自力TLE。。
・にぶたん内でDFS
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

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

def rec(x, di, X):
    if di == 10:
        return 0
    if int(x) > X:
        return 0
    last = int(x[-1])
    res = 0
    res += rec(x+str(last), di+1, X)
    if last != 0:
        res += rec(x+str(last-1), di+1, X)
    if last != 9:
        res += rec(x+str(last+1), di+1, X)
    return res + 1

def check(X):
    res = 0
    for i in range(1, 10):
        res += rec(str(i), 0, X)
    return res >= K

K = INT()

res = bisearch_min(0, 3234566668, check)
print(res)
