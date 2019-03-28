# -*- coding: utf-8 -*-

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

N,Q=MAP()
S=input()

sm=[0]*(N+1)
for i in range(1, N):
    if S[i-1]=='A' and S[i]=='C':
        sm[i+1]=sm[i]+1
    else:
        sm[i+1]=sm[i]

for i in range(Q):
    l,r=MAP()
    print(sm[r]-sm[l])
