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
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    B = LIST()

    A = [0] * (N*2)
    S = set()
    for i in range(N):
        A[i*2] = B[i]
        S.add(B[i])
    
    for i in range(N):
        b = A[i*2]
        for j in range(b+1, N*2+1):
            if j not in S:
                A[i*2+1] = j
                S.add(j)
                break
    
    for i in range(N):
        if A[i*2] > A[i*2+1]:
            print(-1)
            break
    else:
        print(*A)
