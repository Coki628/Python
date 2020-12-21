# -*- coding: utf-8 -*-

import sys

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
M = INT()
A = list(range(1, N*2+1))

def f1(A, k):
    A1 = A[:k]
    A2 = A[k:]
    return A2 + A1

def f2(A):
    res = []
    for i in range(N):
        res.append(A[i])
        res.append(A[i+N])
    return res

for i in range(M):
    q = INT()
    if q == 0:
        A = f2(A)
    else:
        A = f1(A, q)
[print(a) for a in A]
