# -*- coding: utf-8 -*-

"""
・実験用
"""

import sys
from itertools import product
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()

S = set()
for prod in product(range(N), repeat=N):
    ans = [0] * N
    for i, a in enumerate(prod):
        cnt = set()
        for j, b in enumerate(prod):
            if i == j:
                continue
            cnt.add(b)
        ans[i] = len(cnt)
    ans.sort()
    ans = tuple(ans)
    # print(prod)
    # print(ans)
    S.add(ans)
print(sorted(S))
