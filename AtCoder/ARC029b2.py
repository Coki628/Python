# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc029
　　　https://mathwords.net/heimenkaiten
　　　https://atarimae.biz/archives/23930
・数学系、回転行列、ALDS1_5_Cで使ったものを流用。
・斜めの位置を調べるために、座標を回転させる。
・最善の角度を見つけるのには二分探索を使用する。
・行列積の計算に汎用的な関数をABC009dから持ってきた。(TLE)
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

def mat_dot(A, B):
    """ 行列の積 """

    # 1次元リストが来たら2次元の行列にする
    if not isinstance(A[0], list) and not isinstance(A[0], tuple):
        A = [A]
    if not isinstance(B[0], list) and not isinstance(A[0], tuple):
        B = [[b] for b in B]
    n1 = len(A)
    n2 = len(A[0])
    m1 = len(B)
    m2 = len(B[0])
    res=list2d(n1, m2, 0)
    for i in range(n1):
        for j in range(m2):
            for k in range(n2):
                res[i][j] += A[i][k] * B[k][j]
    return res

# 回転行列で座標Bを回転
def rotate(B, dig):
    A = [
        [cos(radians(dig)), -sin(radians(dig))], 
        [sin(radians(dig)), cos(radians(dig))]
    ]
    # (0,0)を軸としてB(x1,y1)を左回りにdig度回転させた座標
    res = mat_dot(A, B)
    return (res[0][0], res[1][0])

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
    x, y = rotate((A, B), m)
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
    x1, _ = rotate((0, B), res)
    # 右下の点を回転させた後のx
    x2, _ = rotate((A, 0), res)
    # 右上の点を回転させた後のy
    _, y = rotate((A, B), res)
    # 最適な向きにしたノートの縦横がc,dに収まるかどうか
    if x2-x1 <= c and y <= d:
        YES()
    else:
        NO()
