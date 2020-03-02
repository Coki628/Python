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

B = LIST()
N = INT()
A = [input() for i in range(N)]

D = {}
for i, b in enumerate(B):
    i, b = str(i), str(b)
    D[b] = i

A2 = [()] * N
for i in range(N):
    a2 = ''
    for s in A[i]:
        a2 += D[s]
    A2[i] = (int(a2), A[i])
A2.sort()
[print(a) for _, a in A2]
