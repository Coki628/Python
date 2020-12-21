# -*- coding: utf-8 -*-

"""
・自力AC！
・しっかり実験すると規則性が見えるやつ。
"""

import sys
from itertools import permutations

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

def check(li):
    while len(li) > 1:
        nxt = []
        for i in range(len(li)-2):
            li2 = [li[i], li[i+1], li[i+2]]
            li2.sort()
            nxt.append(li2[1])
        li = nxt.copy()
    return li[0]

N, X = MAP()
M = N*2 - 1

# C = [0] * (N*2-1)
# for perm in permutations(range(N*2-1)):
#     res = check(list(perm))
#     C[res] += 1
# print(C)

# A = LIST()
# print(check(A))

# 一番端は無理
if X == 1 or X == M:
    No()
    exit()

A = [0] * M
A[M//2] = X
# 欲しい数を中央に、それより弱い(中央から遠い)値を隣に置く
if X <= M//2:
    X2 = X - 1
else:
    X2 = X + 1
A[M//2+1] = X2

# 他はそれ以外を適当に置けばOK
a = 1
for i in range(M):
    while a == X or a == X2:
        a += 1
    if i == M//2 or i == M//2+1:
        continue
    A[i] = a
    a += 1
Yes()
[print(a) for a in A]
