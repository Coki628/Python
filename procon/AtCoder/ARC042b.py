# -*- coding: utf-8 -*-

"""
参考：https://mathtrain.jp/tentotyokusen
　　　https://manapedia.jp/text/2979
・自力AC
・数学系、点と直線の距離
・2点を通る直線の式、とりあえず傾き出してそこから。
・公式に当てはめるために式変形するのが色々大変だった。
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
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

X, Y = MAP()
N = INT()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

mn = INF
for i in range(N):
    x1, y1 = XY[i]
    x2, y2 = XY[(i+1)%N]
    # 垂直
    if x2-x1 == 0:
        a = 1
        b = 0
        c = -x1
    # 水平
    elif y2-y1 == 0:
        a = 0
        b = 1
        c = -y1
    else:
        # 傾きaを出して、ax+by+c=0の形を用意する
        a = (y2-y1) / (x2-x1)
        b = -1
        c = y1 - a*x1
    # 点XYと直線ax+by+c=0の距離
    d = abs(a*X+b*Y+c) / sqrt(a**2+b**2)
    mn = min(mn, d)
print(mn)
