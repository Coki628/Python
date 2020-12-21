# -*- coding: utf-8 -*-

"""
・自力WA
・2点間距離の最大/2でいいかもと思ったけど、さすがにこれだけじゃなかった。
"""

import sys
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

ans = 0
for i in range(N):
    x1, y1 = XY[i]
    for j in range(i+1, N):
        x2, y2 = XY[j]
        ans = max(ans, hypot(x1-x2, y1-y2))
print(ans / 2)
