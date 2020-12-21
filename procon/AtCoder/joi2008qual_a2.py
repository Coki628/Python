# -*- coding: utf-8 -*-

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

for i in range(3):
    h1, m1, s1, h2, m2, s2 = MAP()
    sm1 = h1 * 3600 + m1 * 60 + s1
    sm2 = h2 * 3600 + m2 * 60 + s2
    seg = sm2 - sm1
    h = seg // 3600
    m = (seg % 3600) // 60
    s = (seg % 3600) % 60
    print(h, m, s)
