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

N=INT()

# for i in range(N, 0, -1):
#     print(i, N//i, N%i)

m=N
ans=set()
while m>=sqrt(N):
    i=N//m
    while m>0:
        if i==N%m:
            ans.add(m)
            break
        elif i<N%m:
            break
        else:
            m-=1
    m=N//(i+1)

for m in range(1, int(sqrt(N))+1):
    if N//m==N%m:
        ans.add(m)

print(sum(ans))
