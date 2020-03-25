# -*- coding: utf-8 -*-

"""
・実験用
"""

import sys
from itertools import product

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

def rec(cur):
    if len(cur) == 1:
        return cur[0]
    nxt = []
    for i in range(len(cur)-1):
        nxt.append(abs(cur[i] - cur[i+1]))
    return rec(nxt)

N = INT()
# A = [a-1 for a in list(map(int, input()))]

# ans = rec(A)
# print(ans)

for prod in product([0, 1, 2], repeat=N):
    res = rec(prod)
    print(res, prod)
