# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day2/editorial-B.pdf
・公式解
・ベクトルの外積の正負から、回転方向を判断できる。
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

X,Y=MAP()
A,B=MAP()
sx,sy=MAP()
tx,ty=MAP()

# 川の始まりを起点にしてベクトルと取る
vec_river=(X-0, B-A)
vec_s=(sx-0, sy-A)
vec_t=(tx-0, ty-A)

# 外積の正負で、川から見た回転方向が分かる
cross_s=cross(vec_river, vec_s)
cross_t=cross(vec_river, vec_t)

# xorすると条件スッキリ
if (cross_s>0) ^ (cross_t>0):
    Yes()
else:
    No()
