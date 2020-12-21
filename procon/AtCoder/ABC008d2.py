# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc008
　　　http://mayokoex.hatenablog.com/entry/2016/03/17/123712
　　　https://torus711.hatenablog.com/entry/20140512/1399913264
・dictで管理するメモ化再帰
・区間DPっぽく、区切った狭い領域から決めていく。
・計算量は再帰の遷移数と内ループで(定数入れると)4N*4N=16N^2=16*30^2=14400
　…でももっとある気がする。pythonだけど0.1秒くらいかかってたし。
"""

import sys
from collections import defaultdict

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

W,H=MAP()
N=INT()
XY=[]
for i in range(N):
    x,y=MAP()
    XY.append((x, y))

# 範囲は広いけど遷移数は少ないのでdictでメモを管理
# memo[i][j] := 左下座標がiで右上座標がjの長方形領域の最大値
memo=defaultdict(lambda: defaultdict(lambda: -1))
def rec(ld, ru):
    if memo[ld][ru]!=-1:
        return memo[ld][ru]
    # 現在の長方形位置
    l,d=ld
    r,u=ru
    res=0
    # N箇所切ってみて一番いい場所を探す
    for x, y in XY:
        # 今回の長方形の範囲内かどうか
        if l<=x<=r and d<=y<=u:
            # 切った時取れる金塊(同じ長方形内ならどこでも一緒)
            tmp=(r-l)+(u-d)+1
            # 左上
            if x>1 and y<H:
                tmp+=rec((l, y+1), (x-1, u))
            # 右上
            if x<W and y<H:
                tmp+=rec((x+1, y+1), (r, u))
            # 左下
            if x>1 and y>1:
                tmp+=rec((l, d), (x-1, y-1))
            # 右下
            if x<W and y>1:
                tmp+=rec((x+1, d), (r, y-1))
            res=max(res, tmp)
    # メモして返却
    memo[ld][ru]=res
    return res

print(rec((1, 1), (W, H)))
