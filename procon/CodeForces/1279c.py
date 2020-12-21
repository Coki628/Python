# -*- coding: utf-8 -*-

"""
・自力TLE。。。
・愚直にシミュレーションしたらダメだった。
"""

import sys
from heapq import heappush, heappop

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
    B = LIST()
    B2 = [INF] * (N+1)
    for i, b in enumerate(B):
        B2[b] = i + 1
    for i in range(N):
        a = A[i]
        order = B2[a]
        A[i] = (-order, a)
    A = A[::-1]

    que = []
    cnt = 0
    for b in B:
        while 1:
            order, a = A.pop()
            cnt += 1
            if a == b:
                break
            heappush(que, (order, a))
        while que:
            order, a = heappop(que)
            A.append((order, a))
            cnt += 1
    print(cnt)
