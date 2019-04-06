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

X,Y,Z,K=MAP()
A=sorted(LIST(), reverse=True)
B=sorted(LIST(), reverse=True)
C=sorted(LIST(), reverse=True)

gapA=[]
for i in range(X-1):
    gapA.append(A[i]-A[i+1])
gapB=[]
for i in range(Y-1):
    gapB.append(B[i]-B[i+1])
gapC=[]
for i in range(Z-1):
    gapC.append(C[i]-C[i+1])
gapA.append(INF)
gapB.append(INF)
gapC.append(INF)

ans=A[0]+B[0]+C[0]
print(ans)
i=j=k=l=0
while i<K:
    if j<X and gapA[j]<=gapB[k] and gapA[j]<=gapC[l]:
        ans-=gapA[j]
        j+=1
    elif k<Y and gapB[k]<=gapA[j] and gapB[k]<=gapC[l]:
        ans-=gapB[k]
        k+=1
    elif l<Z and gapC[l]<=gapA[j] and gapC[l]<=gapB[k]:
        ans-=gapC[l]
        l+=1
    print(ans)
    i+=1
