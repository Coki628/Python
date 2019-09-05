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

N, M = MAP()
A = [None] * N
B = [None] * N
for i in range(N):
    A[i] = LIST()
for i in range(N):
    B[i] = LIST()

cur = 0
for i in range(N):
    mn = INF
    for j in range(M):
        nxt = A[i][j] - cur % A[i][j] if cur % A[i][j] else 0
        nxt += B[i][j]
        mn = min(mn, nxt)
    cur += mn
print(cur)
