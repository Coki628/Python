# -*- coding: utf-8 -*-

"""
・終了後すぐくらいに自分の書いた条件の反例見つけて、それに合わせて直したらAC。
　あと10分あればいけたかもしんない。残念。。
"""

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

def check(A, B):
    for i in range(N):
        if A[i] < B[i]:
            return False
    return True

for _ in range(INT()):
    N = INT()
    A = LIST()

    if N % 2 == 0:
        B1 = [0] * N
        B2 = [0] * N
        for i in range(N//2-1):
            B1[i] = B2[i] = i
        B1[N//2-1] = B2[N//2] = N // 2
        B2[N//2-1] = B1[N//2] = N // 2 - 1
        for i in range(N//2+1, N):
            B1[i] = B2[i] = N - i - 1
        if check(A, B1) or check(A, B2):
            Yes()
        else:
            No()
    else:
        B = [0] * N
        for i in range(N//2):
            B[i] = i
        for i in range(N//2, N):
            B[i] = N - i - 1
        if check(A, B):
            Yes()
        else:
            No()
