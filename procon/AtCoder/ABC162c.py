# -*- coding: utf-8 -*-

"""
・3重ループで愚直にGCDする。
・計算量200^3とlogで800万*8くらい？だけどpythonAC1.2秒、pypyAC0.2秒。
　言語アップデートで速くなってるな。
"""

import sys
from math import gcd

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

K = INT()

ans = 0
for i in range(1, K+1):
    for j in range(1, K+1):
        ij = gcd(i, j)
        for k in range(1, K+1):
            ans += gcd(ij, k)
print(ans)
