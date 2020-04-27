# -*- coding: utf-8 -*-

"""
・自力AC！
・二分探索、判定パートを丁寧に作る。
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

# 数列の数m個で条件を満たせるか
def check(m):
    C = [[] for i in range(m)]
    # 大きい方から順にm個の数列に均等に割り振っていく
    i = 0
    for a in A:
        C[i].append(a)
        i = (i+1) % m
    for i in range(m):
        for j, c in enumerate(C[i]):
            # 条件を満たさない箇所があればNG
            if not B[c] >= j+1:
                return False
    return True

N, M = MAP()
A = LIST()
B = [0] + LIST()

A.sort(reverse=1)
res = bisearch_min(0, N+1, check)
ans = [[] for i in range(res)]
i = 0
for a in A:
    ans[i].append(a)
    i = (i+1) % res
print(len(ans))
for i in range(res):
    print(len(ans[i]), *ans[i])
