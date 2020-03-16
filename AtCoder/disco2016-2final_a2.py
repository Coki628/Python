# -*- coding: utf-8 -*-

"""
・hypotで原点からの距離出して半径と比べる方が楽だった。
"""

import sys
from math import hypot

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
        # 原点から(i, j)までの距離
        dist=hypot(i, j)
        # それが半径以内であれば、円の内側にあることになる
        if dist<=r:
            grid[i][j]=True

cnt=0
for i in range(r//C):
    for j in range(r//C):
        # チップを形成する4点が全て範囲内にあるかどうか
        if grid[i*C][j*C] and grid[(i+1)*C][j*C] and grid[i*C][(j+1)*C] and grid[(i+1)*C][(j+1)*C]:
            cnt+=1
# 1/4円で数えたので*4する
print(cnt*4)
