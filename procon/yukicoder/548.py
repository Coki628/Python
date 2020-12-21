# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・集計の集計
"""

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

S = input()

C = Counter(S)
C2 = Counter(C.values())
if C2[1] == 13:
    for c in 'abcdefghijklm':
        print(c)
elif C2[1] == 11 and C2[2] == 1:
    for c in 'abcdefghijklm':
        if C[c] == 0:
            print(c)
            break
else:
    print('Impossible')
