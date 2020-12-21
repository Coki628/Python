# -*- coding: utf-8 -*-

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
Ps=[None]*Q
for i in range(Q):
    l,r=MAP()
    Ps[i]=(l,r)
    imos[l]+=1
    imos[r+1]-=1

for i in range(N+1):
    imos[i+1]+=imos[i]

sm=0
for i in range(1, N+1):
    if imos[i]>0:
        sm+=1

mx=0
for i in range(Q):
    for j in range(i+1, Q):
        l1,r1=Ps[i]
        l2,r2=Ps[j]
        tmp=sm
        cp=copy(imos)
        for k in range(l1, r1+1):
            cp[k]-=1
            if cp[k]==0:
                tmp-=1
        for k in range(l2, r2+1):
            cp[k]-=1
            if cp[k]==0:
                tmp-=1
        mx=max(mx, tmp)
print(mx)
