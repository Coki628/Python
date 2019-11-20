# -*- coding: utf-8 -*-

import sys
from operator import itemgetter

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

for _ in range(INT()):
    N, M = MAP()
    A = LIST()
    A2 = []
    for i, a in enumerate(A):
        A2.append((i+1, a))
    if N > M or N <= 2:
        print(-1)
        continue
    cost = sum(A) * 2
    edges = []
    for i in range(N):
        if i != N-1:
            edges.append((i+1, i+2))
        else:
            edges.append((i+1, i+2-N))
    M -= N
    A2.sort(key=itemgetter(1))
    cost += (A2[0][1]+A2[1][1]) * M
    for i in range(M):
        edges.append((A2[0][0], A2[1][0]))
    print(cost)
    for a, b in edges:
        print(a, b)
