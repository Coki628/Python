# -*- coding: utf-8 -*-

"""
・400点自力AC
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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST(N)

ans = A[0] // 2
mod = A[0] % 2
for i in range(1, N):
    ans += A[i] // 2
    tmpmod = A[i] % 2
    if mod == tmpmod == 1:
        ans += 1
        mod = 0
    elif tmpmod == 1:
        mod = 1
    elif A[i] == 0:
        mod = 0
print(ans)
