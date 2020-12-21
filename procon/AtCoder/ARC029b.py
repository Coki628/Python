# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc029
　　　https://mathwords.net/heimenkaiten
　　　https://atarimae.biz/archives/23930
・数学系、回転行列、ALDS1_5_Cで使ったものを流用。
・斜めの位置を調べるために、座標を回転させる。
・最善の角度を見つけるのには二分探索を使用する。
"""

import sys
from math import sin, cos, radians

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

# 回転行列で座標s,tを回転
def rotate(s, t, dig):
    x1, y1 = s
    x2, y2 = t
    # sを軸としてtを左回りにdig度回転させた座標(x3,y3)
    x3 = x1 + (x2-x1) * cos(radians(dig)) - (y2-y1) * sin(radians(dig))
    y3 = y1 + (x2-x1) * sin(radians(dig)) + (y2-y1) * cos(radians(dig))
    return (x3, y3)

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索(小数用) """
    ok = mn
    ng = mx
    for i in range(100):
        mid = (ok+ng) / 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

# 回転後のノートの高さyがd以内に収まっているかどうか
def calc(m):
    # m度左回りに回転
    x, y = rotate((0, 0), (A, B), m)
    return y <= d

# A:横(x), B:縦(y), と思っておく
A, B = MAP()
# ノートを横長に統一
A, B = max(A, B), min(A, B)
N = INT()
for i in range(N):
    c, d = MAP()
    # 箱を縦長に統一
    c, d = min(c, d), max(c, d)
    # 斜めにしないでOKのケース
    if A <= c and B <= d or A <= d and B <= c:
        YES()
        continue
    # ノートの縦が箱に収まる最大の角度
    res = bisearch_max(0, 90, calc)
    # 左上の点を回転させた後のx
    x1, _ = rotate((0, 0), (0, B), res)
    # 右下の点を回転させた後のx
    x2, _ = rotate((0, 0), (A, 0), res)
    # 右上の点を回転させた後のy
    _, y = rotate((0, 0), (A, B), res)
    # 最適な向きにしたノートの縦横がc,dに収まるかどうか
    if x2-x1 <= c and y <= d:
        YES()
    else:
        NO()
