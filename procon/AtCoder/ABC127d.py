# -*- coding: utf-8 -*-

"""
・400点自力AC
"""

import sys
from operator import itemgetter

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

N,M=MAP()
A=LIST()
BC=[None]*M
for i in range(M):
    b,c=MAP()
    BC[i]=(b, c)

A.sort()
BC.sort(key=itemgetter(1), reverse=True)

idx=0
A2=[0]*N
for i in range(M):
    b,c=BC[i]
    if idx<N:
        for j in range(idx, idx+b):
            if j==N: break
            A2[j]=c
        idx+=b

ans=0
for i in range(N):
    ans+=max(A[i], A2[i])
print(ans)
