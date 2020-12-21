# -*- coding: utf-8 -*-

"""
・直線の直交・並行判定
・ベクトルの内積と外積を使う。
"""

import sys

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

# ベクトルの内積
def dot(a, b):
    x1,y1=a
    x2,y2=b
    return x1*x2+y1*y2

# ベクトルの外積
def cross(a, b):
    x1,y1=a
    x2,y2=b
    return x1*y2-y1*x2

Q=INT()
for _ in range(Q):
    x1,y1,x2,y2,x3,y3,x4,y4=MAP()
    # 2直線のベクトルを出しておく
    # (今回は方向だけ分かればよくて向きはどっちでもいいので、x1-x2かx2-x1かはどっちでもいい。
    # ただしxとyでどっちにするかは揃える。)
    v1=(x1-x2, y1-y2)
    v2=(x3-x4, y3-y4)
    # 直交判定(内積が0)
    if dot(v1, v2)==0:
        print(1)
    # 平行判定(外積が0)
    elif cross(v1, v2)==0:
        print(2)
    else:
        print(0)
