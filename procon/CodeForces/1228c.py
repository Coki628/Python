# -*- coding: utf-8 -*-

"""
・調査のみ。解法には辿り着けず。
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

def factorize(num: int) -> dict:
    from math import sqrt
    from collections import Counter

    d = Counter()
    for i in range(2, int(sqrt(num))+1):
        while num % i == 0:
            num //= i
            d[i] += 1
        if num == 1:
            break
    if num != 1:
        d[num] += 1
    return d

def prime(x):
    return set(factorize(x).keys())

def g(x, p):
    i = 0
    while x % p**(i+1) == 0:
        i += 1
    return p**i

def f(x, y):
    s = prime(x)
    res = 1
    for p in s:
        res *= g(y, p)
        res %= MOD
    return res

X, N = MAP()

ans = 1
for i in range(1, N+1):
    # print(i, f(X, i))
    ans *= f(X, i)
    ans %= MOD
print(ans)
