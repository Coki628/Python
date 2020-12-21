# -*- coding: utf-8 -*-

"""
参考：https://qiita.com/ykob/items/ab7f30c43a0ed52d16f2
・さくっと自力AC
・数学系、線分の交差判定
"""

import sys

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

# 線分ABとCDの交差判定
def is_intersected(ax, ay, bx, by, cx, cy, dx, dy):
    ta = (cx - dx) * (ay - cy) + (cy - dy) * (cx - ax)
    tb = (cx - dx) * (by - cy) + (cy - dy) * (cx - bx)
    tc = (ax - bx) * (cy - ay) + (ay - by) * (ax - cx)
    td = (ax - bx) * (dy - ay) + (ay - by) * (ax - dx)

    return tc * td < 0 and ta * tb < 0

ax,ay,bx,by=MAP()
N=INT()
XY=[]
for i in range(N):
    x,y=MAP()
    XY.append((x, y))

cnt=0
for i in range(N):
    x1,y1=XY[i]
    x2,y2=XY[(i+1)%N]
    if is_intersected(ax, ay, bx, by, x1, y1, x2, y2):
        cnt+=1

print(cnt//2+1)
