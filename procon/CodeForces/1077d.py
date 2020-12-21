# -*- coding: utf-8 -*-

"""
・自力AC
・集計して二分探索
・ここで必要な作れる最大数は単調性ありでにぶたんでいける。
・最大値が分かったら実際にその値に合わせてやってみる、でOK。
"""

import sys
from collections import Counter

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

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

N, K = MAP()
A = LIST()

C = Counter(A)

def check(m):
    res = 0
    for v in C.values():
        res += v // m
    return res >= K

res = bisearch_max(0, 10**6, check)
ans = []
for k, v in C.items():
    if v // res != 0:
        ans += [k] * (v//res)
print(*ans[:K])
