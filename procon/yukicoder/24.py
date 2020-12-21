# -*- coding: utf-8 -*-

"""
・集合をうまく管理すると答えが定まる。
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

N = INT()
S1 = set(range(10))
S2 = set()
for _ in range(N):
    *li, yn = input().split()
    S3 = set(map(int, li))
    if yn == 'YES':
        S1 &= S3
    else:
        S2 |= S3
ans = S1 & (S2 ^ set(range(10)))
print(list(ans)[0])
