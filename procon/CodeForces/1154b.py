# -*- coding: utf-8 -*-

"""
・場合分けを詰める。
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

N = INT()
A = LIST()

S = set(A)
if len(S) >= 4:
    print(-1)
elif len(S) == 3:
    mn = min(S)
    mx = max(S)
    S.remove(mn)
    S.remove(mx)
    mid = list(S)[0]
    if (mn + mx) / 2 == mid:
        print(mx - mid)
    else:
        print(-1)
elif len(S) == 2:
    mn = min(S)
    mx = max(S)
    if (mn + mx) % 2 == 0:
        print(mx - (mn + mx) // 2)
    else:
        print(mx - mn)
else:
    print(0)
