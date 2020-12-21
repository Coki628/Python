# -*- coding: utf-8 -*-

"""
・これはTLE
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
rowsm=[W]*H
colsm=[H]*W
RC=[None]*K
for i in range(K):
    r,c=MAP()
    rowsm[r-1]-=1
    colsm[c-1]-=1
    RC[i]=(r-1, c-1)

mn=INF
row=0
for i in range(H):
    sm=0
    for j in range(H):
        sm+=rowsm[j]*abs(i-j)
    if mn>sm:
        mn=sm
        row=i
mn=INF
col=0
for i in range(W):
    sm=0
    for j in range(W):
        sm+=colsm[j]*abs(i-j)
    if mn>sm:
        mn=sm
        col=i
ans=(get_sum(row)+get_sum(H-row-1))*W+(get_sum(col)+get_sum(W-col-1))*H
for i in range(K):
    r,c=RC[i]
    ans-=abs(row-r)+abs(col-c)

print(ans)
