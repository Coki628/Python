# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-8
・メモ化再帰、座標圧縮
・概要覚えてたのに、+1するかしないかとか1回目の処理をどうするかとかで
　色々悩んだりバグらせたりで時間かかった。。
・自力諦めて昔の見たら、あとメモ化するだけだったじゃん。。
・なんで同じ遷移起きないって思ったんだろう。。。
・一応、テーマが座圧なのでこっちもやっておく。
・dictメモとやってることほぼ変わらないのにMLEでTLE。何故だ。。
・上記、解決。
　全部のメモについて+1,-1した場所を取ってたんだけど、これだと
　最大でN=30*3=90について4次元メモ=90^4=約6500万の領域でメモ作ってた。
　始点は+1だけ、終点は-1だけしか使わないから、それに合わせて作ると
　60^4=約1300万で済む。これだとpython,pypy共にAC。
　でもdictでやるより全然メモリも食うし遅い。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

def compress(A):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(set(A))):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

W, H = MAP()
N = INT()
XY = []
X1, Y1 = [], []
X2, Y2 = [], []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))
    X1 += [x, x+1]
    Y1 += [y, y+1]
    X2 += [x-1, x]
    Y2 += [y-1, y]

# 座標圧縮
X1_zipped, _ = compress(X1 + [1, W])
Y1_zipped, _ = compress(Y1 + [1, H])
X2_zipped, _ = compress(X2 + [1, W])
Y2_zipped, _ = compress(Y2 + [1, H])

# memo[x1][y1][x2][y2] := 左下座標が(x1, y1)で右上座標が(x2, y2)の長方形領域の最大値
memo = list4d(len(X1_zipped), len(Y1_zipped), len(X2_zipped), len(Y2_zipped), -1)
def rec(x1, y1, x2, y2):
    # メモの添字だけ圧縮後の値にする
    zx1, zy1, zx2, zy2 = X1_zipped[x1], Y1_zipped[y1], X2_zipped[x2], Y2_zipped[y2]
    if memo[zx1][zy1][zx2][zy2] != -1:
        return memo[zx1][zy1][zx2][zy2]
    res = 0
    for x3, y3 in XY:
        if x1 <= x3 <= x2 and y1 <= y3 <= y2:
            tmp = (x2-x1+1) + (y2-y1+1) - 1
            # 左下
            tmp += rec(x1, y1, x3-1, y3-1)
            # 右上
            tmp += rec(x3+1, y3+1, x2, y2)
            # 左上
            tmp += rec(x1, y3+1, x3-1, y2)
            # 右下
            tmp += rec(x3+1, y1, x2, y3-1)
            res = max(res, tmp)
    memo[zx1][zy1][zx2][zy2] = res
    return res

ans = 0
for x, y in XY:
    tmp = W + H - 1
    # 左下
    tmp += rec(1, 1, x-1, y-1)
    # 右上
    tmp += rec(x+1, y+1, W, H)
    # 左上
    tmp += rec(1, y+1, x-1, H)
    # 右下
    tmp += rec(x+1, 1, W, y-1)
    ans = max(ans, tmp)
print(ans)
