# -*- coding: utf-8 -*-

"""
・300点なんとか自力AC。時間すげーかかったけどね。
・企業コンのfinalだから点数より辛口目なのだろう…多分。
"""

import sys
from itertools import accumulate

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

# 1からnまでの等差数列の和
def get_sum(n):
    return (1+n)*n//2

H,W,K=MAP()
# 全てある状態で初期化
rowsm=[W]*H
colsm=[H]*W
RC=[None]*K
for i in range(K):
    r,c=MAP()
    # 入力に応じて引いていく
    rowsm[r-1]-=1
    colsm[c-1]-=1
    RC[i]=(r-1, c-1)

# 区間和を使うために累積和を取る
rowacc=[0]+list(accumulate(rowsm))
colacc=[0]+list(accumulate(colsm))

# 1行目に集めた時に必要な移動距離
row=sm=0
for i in range(H):
    sm+=rowsm[i]*i
mn=sm
for i in range(1, H):
    # i+1行目にした時に増える距離-減る距離
    # (最初～i+1行目までの区間和)-(i+2行目～最後までの区間和)
    sm+=rowacc[i]-(rowacc[H]-rowacc[i])
    if mn>sm:
        mn=sm
        # 最小値を取って行のindexを保持
        row=i

# 列でもほぼ同じことをやる
col=sm=0
for i in range(W):
    sm+=colsm[i]*i
mn=sm
col=0
for i in range(1, W):
    sm+=colacc[i]-(colacc[W]-colacc[i])
    if mn>sm:
        mn=sm
        col=i

# 最小が狙えるrowとcolに対して、全て集めた時の値
ans=(get_sum(row)+get_sum(H-row-1))*W+(get_sum(col)+get_sum(W-col-1))*H
for i in range(K):
    r,c=RC[i]
    # 入力で与えられた部分を引いていく
    ans-=abs(row-r)+abs(col-c)
print(ans)
