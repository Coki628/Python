# -*- coding: utf-8 -*-

"""
・しゃくとり
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

N=INT()
A=LIST()

A.sort()
l=r=ans=0
while l<N:
    while r<N and A[r]-A[l]<=5:
        r+=1
    ans=max(ans, r-l)
    if l==r:
        r+=1
    l+=1
print(ans)
