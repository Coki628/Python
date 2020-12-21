# -*- coding: utf-8 -*-

"""
・自力AC
・幾何、誤差、円と正方形
・d3の長さを出す時にちょこっと三角比。
・条件式の判定はちゃんとEPSを使って誤差対策する。
"""

import sys
from math import sqrt

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
EPS = 10 ** -10

d1, d2 = MAP()

d1, d2 = sqrt(d1), sqrt(d2)
d3 = d2 / sqrt(2)
if d1 > d2+EPS or d1+EPS < d3:
    print(0)
elif abs(d1-d2) < EPS or abs(d1-d3) < EPS:
    print(4)
else:
    print(8)
