# -*- coding: utf-8 -*-

"""
・実験用。ここで規則性を見つけられたからACできた。
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

N = INT()

cnt = [0] * (N+1)
for perm in permutations(range(1, N+1)):
    S = set()
    for p in perm:
        cnt[p] += 1
        for i in range(p+1, N+1):
            if i in S:
                break
            else:
                cnt[i] += 1
        S.add(p)
print(cnt[1:])
gap = [0] * N
for i in range(N):
    gap[i] = cnt[i+1] - cnt[i]
print(gap)
rate = [0] * (N-1)
for i in range(N-1):
    rate[i] = gap[i+1] / gap[i]
print(rate)
