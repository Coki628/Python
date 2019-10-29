# -*- coding: utf-8 -*-

"""
・400点自力AC！
・数学っぽいやつで自力通せたのは嬉しい。かなり時間かかったけどね。。
・幾何、二分探索
・基本的な方針は合ってたけど、sin,cosで斜辺求めたりするところとか色々バグらせて時間かかった。
"""

import sys
from math import atan2, degrees, radians, sqrt, sin, cos

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bisearch_max(mn, mx, func):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    for _ in range(100):
        mid = (ok+ng) / 2
        if func(mid):
            # 上を探しに行く
            ok = mid
        else:
            # 下を探しに行く
            ng = mid
    return ok

a, b, x = MAP()
# 体積を面積にしておく
h = b * (x / (a * a * b))
S = a * h

# カップの底をh1の高さにした時、溢れないかどうか
def calc(h1):
    # 傾けた下の地面の長さ
    c = sqrt(a**2 - h1**2)
    # 傾けた角度
    dig1 = degrees(atan2(h1, c))
    # 傾けた反対側の角度
    dig2 = 90 - dig1
    if dig2 == 0:
        return False
    # カップの上の先端の低い方の高さ
    h2 = b * sin(radians(dig2))
    # 底の高い方よりカップの先端が高い位置にある
    if h2 >= h1:
        # 水面の横の長さ
        d = a / cos(radians(dig1))
        # 下の方の面積(三角形)
        S1 = d * h1 / 2
        # この時点でSのが小さければOK
        if S1 >= S:
            return True
        # 上の方の面積(平行四辺形)
        S2 = S - S1
        # 上の方の水の高さ
        h3 = S2 / d
        # 水面の高さがカップの先端より低ければOK
        return h1 + h3 <= h2
    # 底の高い方よりカップの先端が低い位置にある
    else:
        # 溢れるギリギリの水面の長さ
        e1 = b / cos(radians(dig2))
        # 面積と高さから、実際の水面の長さ
        e2 = S / h2 * 2
        return e2 <= e1

h = bisearch_max(0, a, calc)
w = sqrt(a**2 - h**2)
print(degrees(atan2(h, w)))
