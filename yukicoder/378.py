# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・10^18でもlogだから計算量少ないやつ。
"""

import sys
from itertools import accumulate

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
EPS = 10 ** -10

N = INT()

A = []
while N > 0:
    A.append(N)
    N //= 2

acc = [0] + list(accumulate(A))
ans = 0
for i in range(len(A)):
    ans = max(ans, acc[i]+A[i]*2 - acc[-1])
print(ans)
