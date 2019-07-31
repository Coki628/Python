# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc016
　　　http://aozoragakuen.sakura.ne.jp/taiwa/taiwaNch03/senkeikika/node7.html
・数学系、線分の交差判定
・公式解、符号付き面積で判定する。
・符号付き面積はベクトルの外積/2で出せる。
・でもこれはWA。
　多分、反対側にあるけど線分の範囲からは外れてるやつも数えちゃってるんだと思う。
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

# ベクトルの外積
def cross(a, b):
    x1,y1=a
    x2,y2=b
    return x1*y2-y1*x2

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

# 線分ABとCDの交差判定
def is_intersected(ax, ay, bx, by, cx, cy, dx, dy):
    # aから見たb,c,dへ向かうベクトルにする
    v1=sub((bx, by), (ax, ay))
    v2=sub((cx, cy), (ax, ay))
    v3=sub((dx, dy), (ax, ay))
    # 符号付き面積
    S1=cross(v1, v2)/2
    S2=cross(v1, v3)/2
    # b,cが線分の延長線上にある時の例外処理
    if S1==0 or S2==0: return False
    # 面積の正負が違えば、線分から見て逆側なので交差している
    return (S1>0)^(S2>0)

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
