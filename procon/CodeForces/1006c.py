# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・二分探索、累積和
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

# sm1以下になる最小index
def check(m):
    return acc[N] - acc[m] <= sm1

N = INT()
A = LIST()

acc = [0] + list(accumulate(A))
ans = 0
for i in range(N):
    sm1 = acc[i]
    res = bisearch_min(i-1, N, check)
    # そもそもsm1に足りない
    if res < 0:
        break
    # sm3がsm1ぴったりになるならOK
    sm3 = acc[N] - acc[res]
    if sm1 == sm3:
        ans = max(ans, sm1)
print(ans)
