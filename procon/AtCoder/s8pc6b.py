# -*- coding: utf-8 -*-

"""
・300点自力AC
・中央値に着目する。
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

N = INT()
A, B = [], []
for i in range(N):
    a, b = MAP()
    A.append(a)
    B.append(b)

A.sort()
B.sort()
# 中央値
enter = A[N//2]
_exit = B[N//2]

cnt = 0
for i in range(N):
    cnt += abs(enter-A[i])
    cnt += abs(A[i]-B[i])
    cnt += abs(B[i]-_exit)
print(cnt)
