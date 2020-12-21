# -*- coding: utf-8 -*-

import sys
from itertools import accumulate

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

N, K = MAP()
A = [(a, i) for i, a in enumerate(LIST())]

A.sort(reverse=1)
idxs = set()
sm = 0
for i in range(K):
    idxs.add(A[i][1])
    sm += A[i][0]

ans = []
cnt = 0
for i in range(N):
    if i in idxs:
        cnt += 1
        ans.append(cnt)
        cnt = 0
    else:
        cnt += 1
ans[-1] += cnt
print(sm)
print(*ans)
