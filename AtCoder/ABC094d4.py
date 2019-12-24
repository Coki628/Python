# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC！
・真ん中以上と1つ手前、を偶奇を考慮して切り上げと切り捨てで、計4箇所調べた。
"""

import sys
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

N = INT()
A = [0] + LIST()

A.sort()

def check(x):
    idx = bisect_left(A, x)
    if abs(x - A[idx]) < abs(x - A[idx-1]):
        return A[idx], abs(x - A[idx])
    else:
        return A[idx-1], abs(x - A[idx-1])

# 真ん中を調べる
n = A[-1]
r1, dist1 = check(n // 2)
r2, dist2 = check(ceil(n, 2))
if dist1 < dist2:
    r = r1
else:
    r = r2
print(n, r)
