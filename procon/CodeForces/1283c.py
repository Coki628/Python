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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

src = []
dist = []
S = set()
for i, a in enumerate(A):
    if a == 0:
        src.append(i)
    else:
        S.add(A[i]-1)
for i in range(N):
    if i not in S:
        dist.append(i)

M = len(src)
for i in range(M):
    if src[i] == dist[i]:
        dist[i], dist[(i+1)%M] = dist[(i+1)%M], dist[i]
for i in range(M):
    A[src[i]] = dist[i] + 1
print(*A)
