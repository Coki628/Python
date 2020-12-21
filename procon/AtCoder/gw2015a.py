# -*- coding: utf-8 -*-

"""
・ビット全探索
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

A = [25,39,51,76,163,111,128,133,138]

N = len(A)
se = set()
for S in range(1<<N):
    sm = 0
    for i in range(N):
        if S & 1<<i:
            sm += A[i]
    se.add(sm)
    # G問題だけ例外処理
    se.add(sm + 58)
    se.add(sm + 136)

ans = sorted(se)
[print(a) for a in ans]
