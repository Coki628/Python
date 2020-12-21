# -*- coding: utf-8 -*-

"""
・終了後自力AC。こっちもC詰まったのに引っ張られてバチャ中に解けなかった。。
・累積GCD、二分探索
・言語アップデート、pypyでaccumulateの第2引数が使えるようになってて嬉しい。
"""

import sys
from itertools import accumulate
from math import gcd

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

N, Q = MAP()
A = LIST()
B = LIST()

acc = list(accumulate(A, gcd))

for x in B:
    res = bisearch_min(-1, N, lambda m: gcd(x, acc[m]) == 1)
    if res != N:
        print(res + 1)
    else:
        print(gcd(x, acc[-1]))
