# -*- coding: utf-8 -*-

"""
・二分探索、等差数列の和
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

def get_sum(a, d, n):
    return (2*a + (n-1)*d) * n // 2

def check(m):
    return get_sum(1, 1, m) < K
 
for _ in range(INT()):
    N, K = MAP()
    res = bisearch_max(0, N, check)
    a = N - res - 2
    sm = get_sum(1, 1, res)
    b = N - (K - sm)
    ans = ['a'] * N
    ans[a] = ans[b] = 'b'
    print(''.join(ans))
