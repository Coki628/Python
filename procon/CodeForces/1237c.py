# -*- coding: utf-8 -*-

"""
・コンテンスト終了後自力AC
・ソートしてデータの持ち方を工夫。
　ランレングス圧縮っぽく、同じ値毎にまとめる。
・なるべく同一空間ぽい場所にいる同士で組ませて、余りは後回しにする。
・これ通るんだったら参戦中D悩み過ぎないでこっちもっと取り組めばよかったな。。
"""

import sys
from operator import itemgetter

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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
XYZ = []
IDs = {}
for i in range(N):
    x, y, z = MAP()
    XYZ.append((x, y, z))
    # 出力用にIDを保持
    IDs[(x, y, z)] = i + 1

# zの昇順でz毎にxyを持つ
XYZ.sort(key=itemgetter(2))
Z_XY = []
x, y, z = XYZ[0]
XY = [(x, y)]
prevz = z
for i in range(1, N):
    x, y, z = XYZ[i]
    if z == prevz:
        XY.append((x, y))
    else:
        Z_XY.append((prevz, XY))
        XY = [(x, y)]
    prevz = z
Z_XY.append((prevz, XY))

Z_Y_X = []
for z, xys in Z_XY:
    # yの昇順でy毎にxを持つ 
    xys.sort(key=itemgetter(1))
    Y_X = []
    x, y = xys[0]
    X = [x]
    prevy = y
    for i in range(1, len(xys)):
        x, y = xys[i]
        if y == prevy:
            X.append(x)
        else:
            Y_X.append((prevy, X))
            X = [x]
        prevy = y
    Y_X.append((prevy, X))
    Z_Y_X.append((z, Y_X))

ans = []
tmp = []
for z, xys in Z_Y_X:
    for y, xs in xys:
        # xの昇順で同一y,z上のxをペアにしていく
        xs.sort()
        for i in range(0, len(xs)-len(xs)%2, 2):
            x1 = xs[i]
            x2 = xs[i+1]
            ans.append(IDs[(x1, y, z)])
            ans.append(IDs[(x2, y, z)])
    # 同一y,z上で余った分は各y,z上に要素が1つになってから組ませる
    for y, xs in xys:
        if len(xs)%2 == 1:
            x = xs[-1]
            ans.append(IDs[(x, y, z)])
    # それでも余った分は各z上に要素が1つになってからやるので後回し
    if len(ans) % 2 == 1:
        tmp.append(ans.pop())
for i in tmp:
    ans.append(i)

for i in range(0, N, 2):
    a, b = ans[i], ans[i+1]
    print(a, b)
