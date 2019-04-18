# -*- coding: utf-8 -*-

import sys
from math import factorial

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
A=sorted([INT() for i in range(N)])
B=sorted([INT() for i in range(N)])

i=j=cur=cnt=0
L=[]
if A[0]>B[0]:
    cur=1
while i<N and j<N:
    if cur==0 and A[i]<B[j]:
        i+=1
        cnt+=1
    elif cur==1 and A[i]<B[j]:
        i=j
        if cnt:
            L.append(cnt)
        cnt=0
        cur=0
    elif cur==1 and A[i]>B[j]:
        j+=1
        cnt+=1
    elif cur==0 and A[i]>B[j]:
        j=i
        if cnt:
            L.append(cnt)
        cnt=0
        cur=1
if cnt!=0:
    L.append(cnt)

ans=1
for a in L:
    ans=(ans*factorial(a))%MOD
print(ans)
