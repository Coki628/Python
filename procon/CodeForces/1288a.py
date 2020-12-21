# -*- coding: utf-8 -*-

"""
・システス(多分hackのケース)で落ちてWA。。
・三分探索
・途中で平らな場所があるとダメなやつ。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def check(m):
    return m + (ceil(D, m+1))

for _ in range(INT()):
    N, D = MAP()

    low = -1
    hi = 10**10
    while low+2 < hi:
        m1 = (low*2+hi) // 3
        m2 = (low+hi*2) // 3
        res1 = check(m1)
        res2 = check(m2)
        if res1 <= res2:
            hi = m2
        else:
            low = m1
    mn = min(check(m1), check(m2))
    if mn <= N:
        YES()
    else:
        NO()
