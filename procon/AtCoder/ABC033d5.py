# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc033
　　　http://imagingsolution.blog.fc2.com/blog-entry-50.html
　　　https://univ-study.net/arccos/
　　　https://img.atcoder.jp/iroha2019-day2/editorial-B.pdf
　　　https://atcoder.jp/contests/abc033/submissions/3602863
　　　http://nomoreretake.net/2013/10/21/arctan2/
・数学系、座標上の三角形の角度
・EPSで誤差をカバーする。
　今回は二分探索に使うので、左端が欲しい時は負方向に、右端が欲しい時は正方向にずらす。
・計算量はN^2*logN=400万*約11=約4400万、pypyで5.2秒でAC(この問題は制限7秒だった)
・みんなに合わせて、角度出すのにatan2を使ってみる。
　atan2の正負に合わせて、今回は反時計回り方向に見ていくことにする。
"""

import sys
from math import atan2, degrees
from bisect import bisect_left, bisect_right

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
EPS = 10 ** -9

def sub(a, b):
    return (a[0]-b[0], a[1]-b[1])

N=INT()
XY=[]
for i in range(N):
    x,y=MAP()
    XY.append((x, y))

# ∠bacの時計回り側の角度を計算する
def calc(a, b, c):
    # aから見たb,cへ向かうベクトルにする
    v1=sub(b, a)
    v2=sub(c, a)
    x1,y1=v1
    x2,y2=v2
    # atan2を使うと座標上の角度が出しやすい(atan2(y, x)でy=0から見た角度を返す)
    res=degrees(atan2(y2, x2)-atan2(y1, x1))
    if res<0:
        # 負の数なら時計回り側にあるので逆向きにする
        res+=360
    return res

ans=[0]*3
for i in range(N):
    degs=[]
    j=(i+1)%N
    for k in range(i+2, i+N):
        # j,kはNで循環させる
        k%=N
        degs.append(calc(XY[i], XY[j], XY[k]))
    # 二分探索用にソートして番兵と2周目を追加
    degs.sort()
    degs=[0]+degs+[360]+[deg+360 for deg in degs]

    for j in range(N-1):
        # degs[j]から見て90の左端、90の右端、180の左端のindexを取る(EPSで誤差をカバーする)
        idx1=bisect_left(degs, degs[j]+90-EPS)
        idx2=bisect_right(degs, degs[j]+90+EPS)
        idx3=bisect_left(degs, degs[j]+180-EPS)
        # それぞれ差分で直角と鈍角の個数を出す
        ans[1]+=idx2-idx1
        ans[2]+=idx3-idx2
# 全体 - 直角の数 - 鈍角の数 = 鋭角三角形の個数
ans[0]=(N*(N-1)*(N-2)//6)-ans[1]-ans[2]
print(*ans)
