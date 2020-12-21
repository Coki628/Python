"""
・実験用、メモ化再帰
・setが使いたくなったので、このあとC++でAC。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

N, K = MAP()
A = LIST()

memo = [-1] * N
def rec(i):
    if i >= N:
        return 1
    if i in A:
        return 0
    if memo[i] != -1:
        return memo[i]
    res = 1
    res &= rec(i+1) | rec(i+6)
    res &= rec(i+2) | rec(i+5)
    res &= rec(i+3) | rec(i+4)
    memo[i] = res
    return res
res = rec(1)
print(res)
