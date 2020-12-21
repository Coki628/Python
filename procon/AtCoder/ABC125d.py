# -*- coding: utf-8 -*-

"""
・400点自力AC
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

mn=INF
cnt=0
zero=False
for i in range(N):
    mn=min(mn, abs(A[i]))
    if A[i]<0:
        cnt+=1
    if A[i]==0:
        zero=True
    A[i]=abs(A[i])

if zero:
    print(sum(A))
elif cnt%2==0:
    print(sum(A))
else:
    print(sum(A)-mn*2)
