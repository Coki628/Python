# -*- coding: utf-8 -*-

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 998244353

N, K = MAP()
A = LIST()

border = N - K + 1
B = []
sm = 0
for i, a in enumerate(A):
    if a >= border:
        sm += a
        B.append(i)
C = []
for i in range(K-1):
    C.append(B[i+1] - B[i])
cnt = 1
for c in C:
    cnt *= c
    cnt %= MOD
print(sm, cnt)
