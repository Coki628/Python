# -*- coding: utf-8 -*-

"""
・自力TLE。
・内側の処理はにぶたんで高速化、って思ったけど、にぶたん内がO(N)なので全然速くなかった。。
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
# sys.setrecursionlimit(10 ** 9)
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

Q, X = MAP()
A = [a%X for a in LIST(Q)]

A = A[::-1]
C = Counter(A)

def check(m):
    for i in range(m+1):
        if C[i] == mn:
            return False
    return True

mn = min([C[i] for i in range(X)])
base = mn * X
add = bisearch_max(-1, X, check) + 1

ans = [base + add]
for a in A:
    C[a] -= 1
    if C[a] < mn:
        mn -= 1
        base -= X
    add = bisearch_max(-1, X, check) + 1
    ans.append(base + add)
[print(a) for a in ans[::-1][1:]]
