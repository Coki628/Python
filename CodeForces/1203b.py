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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def check1(C):
    for v in C.values():
        if v % 2 != 0:
            return False
    return True

def check2(A):
    A.sort()
    S = set()
    for i in range(0, N*4, 2):
        a, b = A[i], A[N*4-i-1]
        S.add(a * b)
    return len(S) == 1

for _ in range(INT()):
    N = INT()
    A = LIST()

    C = Counter(A)
    if check1(C) and check2(A):
        YES()
    else:
        NO()
