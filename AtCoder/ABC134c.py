# -*- coding: utf-8 -*-

"""
・前後から累積max
・後ろからもaccumulate使った方がちょっと速いくらいぽい。
"""

import sys
from itertools import accumulate

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
A=[INT() for i in range(N)]

A2=[0]+list(accumulate(A, max))+[0]
A3=[0]+list(accumulate(A[::-1], max))[::-1]+[0]
# A3=[0]*(N+2)
# A3[N]=A[-1]
# for i in range(N-2, -1, -1):
#     A3[i+1]=max(A3[i+2], A[i])

for i in range(1, N+1):
    print(max(A2[i-1], A3[i+1]))
