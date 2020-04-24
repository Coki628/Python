# -*- coding: utf-8 -*-

"""
・自力WA。大きい値は貪欲に大きい値に作らせる。(失敗)
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
A = LIST()

A.sort(reverse=1)
stack = A[::-1]
stack.pop()

j = 0
for i in range(N):
    if i == 0:
        m = 1
    else:
        m = 2**(i-1)
    for _ in range(m):
        if stack[-1] >= A[j]:
            No()
            exit()
        else:
            for _ in range(N-i):
                stack.pop()
        j += 1
Yes()
