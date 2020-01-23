# -*- coding: utf-8 -*-

"""
・自力AC
・素因数分解して場合分け。
・NOになる条件とかミスっててWAしまくった。良くない。
・素数2種類の時のNO判定、すっきりまとまったので修正。最初からこれやれてれば。。
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
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

for _ in range(INT()):
    N = INT()

    primes = sorted(factorize(N).items())
    ans = [1] * 3
    if len(primes) == 1:
        k, v = primes[0]
        if v < 6:
            NO()
            continue
        ans[0] = k
        ans[1] = k ** 2
        ans[2] = k ** (v-3)
    elif len(primes) == 2:
        k1, v1 = primes[0]
        k2, v2 = primes[1]
        if v1 + v2 < 4:
            NO()
            continue
        ans[0] = k1
        ans[1] = k2
        ans[2] = k1**(v1-1) * k2**(v2-1)
    else:
        for i, (k, v) in enumerate(primes):
            if i < 2:
                ans[i] = k ** v
            else:
                ans[2] *= k ** v
    YES()
    print(*ans)
