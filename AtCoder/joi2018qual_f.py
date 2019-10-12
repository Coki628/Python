# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-16
・順列、部分点1想定
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
INF = float('inf')
MOD = 10007

N = INT()
A = LIST()

# 国毎ではなく人毎で国を値に持ったリストを作っておく
B = []
for i, a in enumerate(A):
    for _ in range(a):
        B.append(i)
M = len(B)

# 順列全部やる
cnt = 0
for perm in permutations(B):
    for i in range(1, M):
        if abs(perm[i-1] - perm[i]) < 2:
            break
    else:
        cnt += 1
        cnt %= MOD
print(cnt)
