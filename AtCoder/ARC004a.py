# -*- coding: utf-8 -*-

import sys
from math import hypot
from itertools import combinations

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

mx = 0
for comb in combinations(XY, 2):
    x1, y1 = comb[0]
    x2, y2 = comb[1]
    mx = max(mx, hypot(x2-x1, y2-y1))
print(mx)
