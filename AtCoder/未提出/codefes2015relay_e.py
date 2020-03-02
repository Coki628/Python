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

def check(hm1, hm2):
    if hm1[0] < hm2[0]:
        return True
    elif hm1[0] == hm2[0] and hm1[1] <= hm2[1]:
        return True
    else:
        return False

limit = LIST()
h2, m2 = MAP()

h3 = (h2+6) % 12
m3 = (m2+30) % 60
h4 = (h2+6) % 12
m4 = m2 % 60

if check((h2, m2), limit) or check((h3, m3), limit) or check((h4, m4), limit):
    Yes()
else:
    No()
