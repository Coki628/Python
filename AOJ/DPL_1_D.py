# -*- coding: utf-8 -*-

"""
・LIS(最長増加部分列)
"""

import sys
from bisect import bisect_left

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

N=INT()
A=[INT() for i in range(N)]

B=[A[0]]
for i in range(1, N):
    if A[i]>B[-1]:
        B.append(A[i])
    else:
        B[bisect_left(B, A[i])]=A[i]
print(len(B))
