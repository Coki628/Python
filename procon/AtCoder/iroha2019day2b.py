# -*- coding: utf-8 -*-

"""
参考：http://www.ftext.org/text/subsubsection/2365
・300点自力AC
・式変形
・2点を通る直線のy=ax+bの式は
　y=x*(y2-y1)/(x2-x1)-x1*y2/(x2-x1)+x1*y1/(x2-x1)+y1
・思ったより長ったらしくなって疲れた…。
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

X,Y=MAP()
A,B=MAP()
sx,sy=MAP()
tx,ty=MAP()

x1,y1=0,A
x2,y2=X,B

# 川のy=ax+bの式を出しておく
a=(y2-y1)/(x2-x1)
b=-(x1*y2/(x2-x1))+x1*y1/(x2-x1)+y1

# sとtのx位置を当てはめて、川とs,tのy位置の関係を調べる
river1y=a*sx+b
river2y=a*tx+b

# s,tが川より上にあるか下にあるか
if river1y<sy:
    s='N'
else:
    s='S'
if river2y<ty:
    t='N'
else:
    t='S'

if s!=t:
    Yes()
else:
    No()
