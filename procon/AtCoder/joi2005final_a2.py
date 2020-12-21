# -*- coding: utf-8 -*-

"""
・numpyなしでも全然通った。
　よく見たら制約100万グリッドじゃなくて10万グリッドだ。
"""

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

N, M = MAP()
grid = list2d(N, M, 0)
for i in range(N):
    grid[i] = LIST()

A = [()] * M
for j in range(M):
    sm = 0
    for i in range(N):
        sm += grid[i][j]
    A[j] = (sm, j + 1)

A.sort(key=itemgetter(0), reverse=1)
_, ans = zip(*A)
print(*ans)
