# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・集計して、ソートをちゃんとやる。
"""

import sys
from collections import Counter
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
C = Counter()
for i in range(N):
    INT()
    M, p = MAP()
    S = input().split()
    for s in S:
        C[s] += p

C = list(C.items())
C.sort(key=itemgetter(0))
C.sort(key=itemgetter(1), reverse=1)
for i in range(min(len(C), 10)):
    print(C[i][0], C[i][1])
