# -*- coding: utf-8 -*-

"""
・AGC300点自力AC
・ランレングス圧縮、区間長の大きい方から優先して埋め。
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
INF = float('inf')
MOD = 10 ** 9 + 7

S = input()
N = len(S) + 1

A = []
s = 0
for i in range(N-2):
    if S[i] != S[i+1]:
        A.append((i+1-s, S[i], s, i+1))
        s = i + 1
A.append((N-1-s, S[-1], s, N-1))

A.sort(key=itemgetter(0), reverse=True)

B = [-1] * N
for _, s, l, r in A:
    cnt = 0
    if s == '<':
        for i in range(l, r+1):
            if B[i] == -1:
                B[i] = cnt
            cnt += 1
    else:
        for i in range(r, l-1, -1):
            if B[i] == -1:
                B[i] = cnt
            cnt += 1
print(sum(B))
