# -*- coding: utf-8 -*-

"""
・等差数列の和、二分探索
・絶対計算O(1)でいけるんだろうなぁとか思いながら二分探索書いた。
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

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索 """

    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

def get_sum(a, b, c):
    """ 等差数列の和：(初項a, 末項b, 項数c) """
    return (a+b) * c // 2

y, m = MAP()

base = 2013
m -= 1
cnt = (y-base)*12 + m

res = bisearch_max(0, INF, lambda m: get_sum(13, 12+m, m) <= cnt)
sm = get_sum(13, 12+res, res)
y2 = base + res
m2 = cnt - sm
print(y2, m2+1)
