# -*- coding: utf-8 -*-

"""
・300点自力AC
・でも考察時間かかりすぎたー…。
・円の方程式 (x-a)^2+(y-b)^2=r^2 から、(x-r)^2+(y-r)^2=r^2
　これを変形して x=+-√(r^2-(y-r)^2)+r, y=+-√(r^2-(x-r)^2)+r
　これでxとyの範囲を探せる。
"""

import sys
from math import sqrt

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

def calc(K, L):

    r=L/K/2
    # 切り取るチップの大きさに応じてグリッドを作成
    grid=list2d(int(r*2+1), int(r*2+1), False)
    for i in range(int(r*2)+1):
        for j in range(int(r*2)+1):
            # 円の方程式から、xとyについて解く
            x1=-sqrt(r**2-(j-r)**2)+r
            x2=sqrt(r**2-(j-r)**2)+r
            y1=-sqrt(r**2-(i-r)**2)+r
            y2=sqrt(r**2-(i-r)**2)+r
            # x軸、y軸ともに2点の範囲内にあるか
            if x1<=i<=x2 and y1<=j<=y2:
                grid[i][j]=True
    cnt=0
    for i in range(int(r*2)):
        for j in range(int(r*2)):
            # チップを形成する4点が全て範囲内にあるかどうか
            if grid[i][j] and grid[i+1][j] and grid[i][j+1] and grid[i+1][j+1]:
                cnt+=1
    return cnt

K=INT()

print(calc(K, 200), calc(K, 300))
