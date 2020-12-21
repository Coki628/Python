# -*- coding: utf-8 -*-

import sys
from math import sqrt

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

N,D=MAP()
X=[]
for i in range(N):
    X.append(LIST())

cnt=0
for i in range(N):
    for j in range(i+1, N):
        sm=0
        A=X[i]
        B=X[j]
        for k in range(D):
            sm+=(A[k]-B[k])**2
        ans=sqrt(sm)
        if ans>=0 and ans.is_integer():
            cnt+=1
print(cnt)
