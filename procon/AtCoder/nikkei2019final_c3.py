# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/nikkei2019-final/editorial.pdf
・集めるのに最適なマスは中央値取るだけでいけるようなのでやってみる。
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
# 残っている駒の個数
cnt=H*W-K

row=sm=0
for i in range(H):
    sm+=rowsm[i]
    # 累積和が駒の総数の半分以上になったところが中央値
    if sm>=cnt/2:
        row=i
        break
col=sm=0
for i in range(W):
    sm+=colsm[i]
    if sm>=cnt/2:
        col=i
        break

# 最小が狙えるrowとcolに対して、全て集めた時の値
ans=(get_sum(row)+get_sum(H-row-1))*W+(get_sum(col)+get_sum(W-col-1))*H
for i in range(K):
    r,c=RC[i]
    # 入力で与えられた部分を引いていく
    ans-=abs(row-r)+abs(col-c)
print(ans)
