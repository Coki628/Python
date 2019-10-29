# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/kyopro_friends/status/1188451724278677504
・人様の解説参考でスッキリとまとめようとした。
・場合分けは常に水が半分以上か以下かでOK(考えてみりゃそれはそう)
・水が多い時の判定は、全体 - 上の空間 って考え方にすると、注目するのが三角形1個で済む。
・しかもこれもよく見ると、下の浮かせてる空間の大きさと一致。同じ値が使える。図が分かりやすい。
"""

import sys
from math import degrees, radians, sin, cos

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

# カップをdig1度傾けた時、溢れるかどうか
def calc1(dig1):
    # 傾けた反対側の角度
    dig2 = 90 - dig1
    if dig2 == 0:
        return False
    # カップの上の先端の低い方の高さ
    h = b * sin(radians(dig2))
    # 溢れるギリギリの水面の横の長さ
    c = b / cos(radians(dig2))
    # 溢れるギリギリの面積
    limit = h * c / 2
    return S <= limit

def calc2(dig1):
    # コップの底を持ち上げる高さ
    h = a * sin(radians(dig1))
    # 溢れるギリギリの水面の横の長さ
    c = a / cos(radians(dig1))
    # 溢れるギリギリの面積(全体 - 上の空間)
    limit = a * b - h * c / 2
    return S <= limit

a, b, x = MAP()

# 体積を面積にしておく
S = x / a
# 水の量が全体の半分以上か以下かで場合分け
if S <= a * b / 2:
    calc = calc1
else:
    calc = calc2
dig = bisearch_max(0, 90, calc)
print(dig)
