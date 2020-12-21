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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST()

    mx = 0
    S = set()
    for i in range(N-1):
        if A[i] != -1 and A[i+1] != -1:
            mx = max(mx, abs(A[i] - A[i+1]))
        if A[i] != -1 and A[i+1] == -1:
            S.add(A[i])
        if A[i] == -1 and A[i+1] != -1:
            S.add(A[i+1])
    
    if S:
        li = sorted(S)
        k = (li[0] + li[-1]) // 2
        m = max(max(li[-1]-k, k-li[0]), mx)
        print(m, k)
    else:
        print(mx, 0)
