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

def check(A):
    S = set()
    div = False
    x = -1
    for i, a in enumerate(A):
        if a in S:
            if div or max(S) != i:
                return -1
            else:
                div = True
                x = i
                S = set([a])
        else:
            S.add(a)
    if x + max(S) == N:
        return x
    else:
        return -1

for _ in range(INT()):
    N = INT()
    A = LIST()

    res1 = check(A)
    res2 = check(A[::-1])
    if res2 != -1:
        res2 = N - res2
    if res1 != -1 and res2 != -1:
        if res1 != res2:
            print(2)
            print(res1, N-res1)
            print(res2, N-res2)
        else:
            print(1)
            print(res1, N-res1)
    elif res1 != -1:
        print(1)
        print(res1, N-res1)
    elif res2 != -1:
        print(1)
        print(res2, N-res2)
    else:
        print(0)
