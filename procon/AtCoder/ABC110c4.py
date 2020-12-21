# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・苦戦しすぎた。。WA叩きまくった後、なんとかAC。
・色々難しく考えない方が良かった。最終的にサンプルからエスパーしただけみたいになってしまったけど。。
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
T = input()

C1 = sorted(Counter(S).values())
C2 = sorted(Counter(T).values())

N = len(C1)
if len(C2) != N:
    No()
    exit()
for i in range(N):
    if C1[i] != C2[i]:
        No()
        exit()
Yes()
