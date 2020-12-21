# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2020/01/12/224200
　　　https://mathtrain.jp/goshin
・幾何、外接円の中心(外心)
・答えになりうる円の中心座標を全列挙する。
・必要なのは2点間の中心と3点間の外接円の中心。
・後者は公式から出るけど、四則計算がベクトルなのか数値なのかで処理が変わってくるから、よく確認してやる。
・最小包含円ライブラリで一撃、ってのもあるようなのでそのうち。
"""

import sys
from math import hypot, sin, atan2, degrees, radians

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def add(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1+x2, y1+y2)

def sub(a, b):
    x1, y1 = a
    x2, y2 = b
    return (x1-x2, y1-y2)

def mul(a, b):
    x1, y1 = a
    if isinstance(b, tuple):
        x2, y2 = b
    else:
        x2, y2 = b, b
    return (x1*x2, y1*y2)

def div(a, b):
    x1, y1 = a
    if isinstance(b, tuple):
        x2, y2 = b
    else:
        x2, y2 = b, b
    return (x1/x2, y1/y2)

def get_dig(a, b, c):
    # aから見たb,cへ向かうベクトル
    v1 = sub(b, a)
    x1, y1 = v1
    v2 = sub(c, a)
    x2, y2 = v2
    # atan2がx=0から見た角度を返すので、差分を取る
    res = degrees(atan2(y2, x2) - atan2(y1, x1))
    # 今回は向き関係なく小さい方の角度が欲しい
    res = abs(res)
    if res >= 180:
        res = 360 - res 
    return res

def get_dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return hypot(x1-x2, y1-y2)

# 3点から外接円の中心(外心)を求める
def calc(a, b, c):
    dig1 = get_dig(a, b, c)
    dig2 = get_dig(b, c, a)
    dig3 = get_dig(c, a, b)
    try:
        # 外心を求める公式から
        return div(add(add(mul(a, sin(radians(dig1)*2)), mul(b, sin(radians(dig2)*2))), mul(c, sin(radians(dig3)*2))),
            sin(radians(dig1)*2) + sin(radians(dig2)*2) + sin(radians(dig3)*2))
    except:
        # 3点が三角形を作らないなら、2点間距離で考慮できてるはずなので無視
        return ()

N = INT()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

centers = []
for i in range(N):
    for j in range(i+1, N):
        # 2点間
        xy = div(add(XY[i], XY[j]), 2)
        centers.append(xy)

for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            # 3点間
            xy = calc(XY[i], XY[j], XY[k])
            if xy:
                centers.append(xy)

# 全列挙した円の中心から、半径を最小にできるものを選ぶ
ans = INF
for center in centers:
    mx = 0
    for xy in XY:
        mx = max(mx, get_dist(center, xy))
    ans = min(ans, mx)
print(ans)
