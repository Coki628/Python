# -*- coding: utf-8 -*-

import sys
from collections import Counter
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, M = MAP()
    S = [ord(s)-97 for s in input()]
    A = LIST()

    imos = [0] * (N+1)
    imos[0] += 1
    imos[N] -= 1
    for a in A:
        imos[0] += 1
        imos[a] -= 1
    imos = list(accumulate(imos))

    ans = [0] * 26
    for i, s in enumerate(S):
        ans[s] += imos[i]
    print(*ans)
