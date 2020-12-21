# -*- coding: utf-8 -*-

"""
・これはWA
・交点の数じゃなくて、交点のある列数だと思ってた。。
"""

import sys

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

for i in range(INT()):
    N = INT()
    A = [-a for a in LIST()]
    M = INT()
    B = LIST()
    A1, A2 = [], []
    for a in A:
        if a % 2 == 0:
            A1.append(a)
        else:
            A2.append(a)
    B1, B2 = [], []
    for a in B:
        if a % 2 == 0:
            B1.append(a)
        else:
            B2.append(a)
    ans = set()
    for a in A1:
        for b in B1:
            ans.add((a + b) // 2)
    for a in A2:
        for b in B2:
            ans.add((a + b) // 2)
    print(len(ans))
