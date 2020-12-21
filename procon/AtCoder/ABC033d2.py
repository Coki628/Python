# -*- coding: utf-8 -*-

"""
参考：http://imagingsolution.blog.fc2.com/blog-entry-50.html
　　　https://univ-study.net/arccos/
　　　https://www.slideshare.net/chokudai/abc033
・数学系、座標上の三角形の角度
・愚直なので部分点想定
・実は鋭角・直角・鈍角は内積だけでもっと簡単に判定できた。(AOJのCGL_2_Aの並行判定とほぼ同じ)
"""

import sys
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

# ベクトルの内積
def dot(a, b):
    x1,y1=a
    x2,y2=b
    return x1*x2+y1*y2

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

N=INT()
XY=[]
for i in range(N):
    x,y=MAP()
    XY.append((x, y))

def calc(a, b, c):
    # aから見たb,cへ向かうベクトルにする
    v1=sub(b, a)
    v2=sub(c, a)
    # 内積
    return dot(v1, v2)

ans=[0]*3
for comb in combinations(XY, 3):
    a,b,c=comb
    # 内積の値から鋭角・直角・鈍角は判定できる
    mn=min(calc(a, b, c), calc(b, c, a), calc(c, a, b))
    if mn<0:
        ans[2]+=1
    elif mn==0:
        ans[1]+=1
    else:
        ans[0]+=1

print(*ans)
