# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/blog/entry/66062
・式変形：総ダメージ量 >= HP となるX(最後までやるラウンド数)
　sm*X+mx >= H
　X >= (H-mx)/sm
　ラウンド数は整数なので、/smを切り上げて、最後の(途中で終わる)ラウンドは愚直すればいい。
"""

import sys

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

H,N=MAP()
D=LIST()

mx=-INF
sm=0
for i in range(N):
    # ダメージ量を+にして考えた方が後で色々やりやすい
    sm-=D[i]
    mx=max(mx, sm)
    # 初回ラウンド中に終了するケース
    if mx>=H:
        print(i+1)
        exit()

# いつまでも終わらないケース
if sm<=0:
    print(-1)
    exit()

# 式変形で最後までやるラウンド数Xを出す
X=ceil(H-mx, sm)
H-=X*sm
ans=X*N

# 途中で終わる最終ラウンドは愚直にやる
for i in range(N):
    H+=D[i]
    ans+=1
    if H<=0:
        print(ans)
        break
