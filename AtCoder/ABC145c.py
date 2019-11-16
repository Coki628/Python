# -*- coding: utf-8 -*-

"""
・300点自力AC
・制約がN<=8なのでシンプルに順列の全列挙。
・python,pypy共にAC0.2秒。
"""

import sys
from itertools import permutations
from math import hypot

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
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

sm = cnt = 0
for perm in permutations(XY):
    dist = 0
    prev = (0, 0)
    for i, (x, y) in enumerate(perm):
        if i == 0:
            prev = (x, y)
        else:
            x2, y2 = prev
            dist += hypot(x-x2, y-y2)
    sm += dist
    cnt += 1
print(sm/cnt)
