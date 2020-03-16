# -*- coding: utf-8 -*-

"""
・二分探索
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

N, T = MAP()
AB = []
total = 0
for i in range(N):
    a, b = MAP()
    total += a
    AB.append(a - b)

# 時間を減らせる度合いの大きいもの順にしておく
AB.sort(reverse=1)

def check(m):
    # m個目の宿題まで写す
    sm = sum(AB[:m])
    # 全体 - 減少分 でT以内となるか
    return total - sm <= T

res = bisearch_min(-1, N+1, check)
if res == N+1:
    print(-1)
else:
    print(res)
