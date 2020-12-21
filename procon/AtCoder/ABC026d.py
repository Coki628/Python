# -*- coding: utf-8 -*-

"""
・D自力AC
・この辺じゃ珍しくちゃんと想定解
・小数で二分探索
・この関数は単調増加じゃないらしいけど、どの解でもいいならこれでOKみたい。
"""

import sys
from math import pi, sin

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

A,B,C=MAP()

def f(t):
    return A*t+B*sin(C*t*pi)

def calc(mid):
    return f(mid)>=100

low=0
hi=100000
for i in range(100000):
    mid=(low+hi)/2
    if calc(mid):
        hi=mid
    else:
        low=mid
print(hi)
