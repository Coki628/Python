# -*- coding: utf-8 -*-

import sys
from collections import Counter

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

N,K=MAP()
A=LIST()

C=Counter()
for i in range(N):
    C[A[i]%K]+=1

ans=C[0]//2
del C[0]
if K%2==0:
    ans+=C[K//2]//2
    del C[K//2]

for k in sorted(C.keys()):
    if k>K//2: 
        break
    ans+=min(C[k], C[K-k])

print(ans*2)
