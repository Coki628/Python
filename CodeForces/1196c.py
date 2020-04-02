# -*- coding: utf-8 -*-

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

for _ in range(INT()):
    N = INT()

    L = D = -INF
    R = U = INF
    for _ in range(N):
        x, y, l, u, r, d = MAP()
        if not l:
            L = max(L, x)
        if not u:
            U = min(U, y)
        if not r:
            R = min(R, x)
        if not d:
            D = max(D, y)
    if L > R or D > U:
        print(0)
        continue
    ans = [0] * 3
    ans[0] = 1
    if L == -INF and R == INF:
        ans[1] = 0
    elif L == -INF:
        ans[1] = R
    else:
        ans[1] = L
    if D == -INF and U == INF:
        ans[2] = 0
    elif D == -INF:
        ans[2] = U
    else:
        ans[2] = D
    print(*ans)
