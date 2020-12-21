# -*- coding: utf-8 -*-

"""
参考：submission:50862938
・2人抜けた状態の最大区間を見つける話。
・1人目が抜けた状態で残り1人になっている場所を累積和しておけば、
　2人目の区間内にそれがいくつ該当するかで、2人目によって0になる場所の数が
　O(1)で分かる。
"""

import sys
from copy import copy

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

N,Q=MAP()

imos=[0]*(N+2)
Pts=[None]*Q
for i in range(Q):
    l,r=MAP()
    Pts[i]=(l,r)
    imos[l]+=1
    imos[r+1]-=1
for i in range(N+1):
    imos[i+1]+=imos[i]

mx=0
for i in range(Q):
    cp=copy(imos)
    l,r=Pts[i]
    # まず1人目の区間を引く
    for j in range(l, r+1):
        cp[j]-=1
    sm=0
    cnt1=[0]*(N+2)
    for j in range(1, N+1):
        # 1人抜けた状態で塗れる区間をカウントしておく
        if cp[j]>0:
            sm+=1
        # 残り1人になっている場所で累積和を取る
        if cp[j]==1:
            cnt1[j]+=1
        cnt1[j+1]+=cnt1[j]
    for j in range(i+1, Q):
        l2,r2=Pts[j]
        # 合計から、区間和を使って2人目の分を引く
        mx=max(mx, sm-(cnt1[r2]-cnt1[l2-1]))

print(mx)
