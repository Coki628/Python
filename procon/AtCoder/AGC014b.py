# -*- coding: utf-8 -*-

"""
・500点自力AC！
・ちょっと考察時間かかったけど、ほぼ想定解の考えが導けた。
　まあ、もしかして、って予想からいくつか試して確信に近づけただけだけど。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
C = Counter()
for i in range(M):
    a, b = MAP()
    C[a] += 1
    C[b] += 1

for k, v in C.items():
    if v%2 == 1:
        NO()
        exit()
YES()
