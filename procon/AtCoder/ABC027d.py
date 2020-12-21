# -*- coding: utf-8 -*-

"""
・色々考えた末、結局部分点狙いの愚直だけどこれもWAっていうね。。
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

S=input()

N=S.count('M')
A=[0]*(N+1)
mcnt=0
for s in S:
    if s=='M':
        mcnt+=1
    elif s=='+':
        A[mcnt]+=1
    else:
        A[mcnt]-=1

idx1=N//2
idx2=N
idx3=N
mx=-INF
for i in range(N//2+1):
    cur=sm=0
    for j in range(N):
        if j<idx1 or idx3<=j:
            cur-=1
        else:
            cur+=1
        sm+=A[j+1]*cur
    mx=max(mx, sm)
    idx1-=1
    idx2-=2
    idx3-=1
idx1=N//2
idx2=N
idx3=N
for i in range(N//2+1):
    cur=sm=0
    for j in range(N):
        if j<idx1 or idx3<=j:
            cur+=1
        else:
            cur-=1
        sm+=A[j+1]*cur
    mx=max(mx, sm)
    idx1-=1
    idx2-=2
    idx3-=1
print(mx)
