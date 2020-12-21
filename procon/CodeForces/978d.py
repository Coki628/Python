# -*- coding: utf-8 -*-

"""
・詰まるところ、最初の2つが決まるとその先の位置関係は確定してしまうので、
　最初の2つだけ+-1して計6通りをうまくいくか試せばOK。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

if N <= 2:
    print(0)
    exit()

def check(a, b):
    dist = b - a
    cur = b
    res = abs(a - A[0]) + abs(b - A[1])
    for i in range(2, N):
        cur += dist
        if cur < A[i] - 1 or cur > A[i] + 1:
            return INF
        res += abs(cur - A[i])
    return res

ans = INF
for i in range(A[0]-1, A[0]+2):
    for j in range(A[1]-1, A[1]+2):
        ans = min(ans, check(i, j))
if ans == INF:
    print(-1)
else:
    print(ans)
