# -*- coding: utf-8 -*-

"""
・尺取法
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

N=INT()
A=LIST()

l=r=0
ans=sm=0
# 外ループで左を動かす
while l<N:
    # 内ループは条件を満たす限り右を動かす
    while r<N and sm+A[r]<=N:
        sm+=A[r]
        r+=1
    if sm==N:
        ans+=1
    if l==r:
        # 左が右に追いついたら、右も左に合わせて+1する
        r+=1
    else:
        # 左を動かす分、合計から引く
        sm-=A[l]
    l+=1
print(ans)
