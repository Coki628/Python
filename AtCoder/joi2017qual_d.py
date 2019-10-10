# -*- coding: utf-8 -*-

"""
・TLE想定、部分点獲得
"""

import sys
from itertools import combinations

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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST(N)

ans = INF
for i in range(1, N):
    for comb in combinations(range(1, N), i):
        comb = [0] + list(comb) + [N]
        mn, mx = INF, -INF
        for j in range(i+1):
            sm = sum(A[comb[j]:comb[j+1]])
            mn = min(mn, sm)
            mx = max(mx, sm)
        dist = mx - mn
        ans = min(ans, dist)
print(ans)
