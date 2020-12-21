# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC
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

N = INT()
A = LIST()

has_zero = False
cnt = 0
for a in A:
    if a < 0:
        cnt += 1
    elif a == 0:
        has_zero = True

A2 = [abs(a) for a in A]
if has_zero or cnt % 2 == 0:
    print(sum(A2))
else:
    print(sum(A2) - min(A2) * 2)
