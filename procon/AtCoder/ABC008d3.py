# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc008
　　　http://mayokoex.hatenablog.com/entry/2016/03/17/123712
　　　https://torus711.hatenablog.com/entry/20140512/1399913264
・dictで管理するメモ化再帰
・区間DPっぽく、区切った狭い領域から決めていく。
・計算量は再帰の遷移数と内ループで(定数入れると)4N*4N=16N^2=16*30^2=14400
　…でももっとある気がする。pythonだけど0.1秒くらいかかってたし。
・座標圧縮でやる版。こうするとdictじゃなくてlistでメモ持っておける。
　でも速度ちょい遅でメモリもこっちのが食ってたから、とりあえずメリットないかも。。
　実装も長方形の範囲1ずらすのを直すの結構時間かかったし。
"""

import sys

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

W,H=MAP()
N=INT()
XY=[]
X, Y = [0]*N, [0]*N
for i in range(N):
    x,y=MAP()
    XY.append((x, y))
    X[i], Y[i] = x, y

# 座標圧縮：圧縮前の値→圧縮後のindexを変換できるようにしておく
zip_x, zip_y = {}, {}
X2 = sorted(set(X+[0, W+1]))
Y2 = sorted(set(Y+[0, H+1]))
for i in range(len(X2)):
    zip_x[X2[i]] = i
for i in range(len(Y2)):
    zip_y[Y2[i]] = i

# memo[i][j][k][l] := 左下座標が(i, j)で右上座標が(k, l)の長方形領域の最大値
memo=list4d(len(X2), len(Y2), len(X2), len(Y2), -1)
def rec(ld, ru):
    # 現在の長方形位置
    l,d=ld
    r,u=ru
    # 座標圧縮
    zl, zd, zr, zu = zip_x[l], zip_y[d], zip_x[r], zip_y[u]
    if memo[zl][zd][zr][zu]!=-1:
        return memo[zl][zd][zr][zu]
    res=0
    # N箇所切ってみて一番いい場所を探す
    for x, y in XY:
        # 今回の長方形の範囲内かどうか
        if l<x<r and d<y<u:
            # 切った時取れる金塊(同じ長方形内ならどこでも一緒)
            tmp=(r-l-1)+(u-d-1)-1
            # 左上
            if x>=1 and y<=H:
                tmp+=rec((l, y), (x, u))
            # 右上
            if x<=W and y<=H:
                tmp+=rec((x, y), (r, u))
            # 左下
            if x>=1 and y>=1:
                tmp+=rec((l, d), (x, y))
            # 右下
            if x<=W and y>=1:
                tmp+=rec((x, d), (r, y))
            res=max(res, tmp)
    # メモして返却
    memo[zl][zd][zr][zu]=res
    return res

print(rec((0, 0), (W+1, H+1)))
