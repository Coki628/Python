# -*- coding: utf-8 -*-

"""
・300点自力AC
・円の方程式を使う。x^2+y^2=r^2 から、x=+-√(r^2-y^2), y=+-√(r^2-x^2)
・基本方針はdisco2017final_aとほぼ一緒。
・円の中心が原点でいける分こっちのがやりやすかったと思う。
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

r,C=MAP()

grid=list2d(110, 110, False)

for i in range(0, r+1, C):
    for j in range(0, r+1, C):
        # 円の方程式から、xとyについて解く
        x1=-sqrt(r**2-j**2)
        x2=sqrt(r**2-j**2)
        y1=-sqrt(r**2-i**2)
        y2=sqrt(r**2-i**2)
        # x軸、y軸ともに2点の範囲内にあるか
        if x1<=i<=x2 and y1<=j<=y2:
            grid[i][j]=True

cnt=0
for i in range(r//C):
    for j in range(r//C):
        # チップを形成する4点が全て範囲内にあるかどうか
        if grid[i*C][j*C] and grid[(i+1)*C][j*C] and grid[i*C][(j+1)*C] and grid[(i+1)*C][(j+1)*C]:
            cnt+=1
# 1/4円で数えたので*4する
print(cnt*4)
