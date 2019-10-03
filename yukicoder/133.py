# -*- coding: utf-8 -*-

"""
・蟻本演習
・N!の全列挙
・yukicoderデビュー
"""

import sys
from itertools import permutations
from math import factorial

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

N = INT()
A = LIST()
B = LIST()

win = 0
for a in permutations(A):
    for b in permutations(B):
        cnt = 0
        for i in range(N):
            if a[i] > b[i]:
                cnt += 1
        if cnt > N / 2:
            win += 1
print(win/factorial(N)**2)
