# -*- coding: utf-8 -*-

"""
・実験から、NかN-1の約数しかなさそうだなー、を察する。
・Nで割り切れないKの時はひたすらKを引き続けるだけなので、
　最終的に N % K が残る。よって N -= K は N %= K とする。
・これで1つずつ調べるのが間に合う。
・うーん、、コンテスト中は見る時間なかったのはあるけど、
　こんな感じでいいならさっさと実験書けばよかったなー、ってのはある。
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

def divisors(N: int) -> set:
    from math import sqrt

    s = {1, N}
    for i in range(2, int(sqrt(N))+1):
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

def check(n, k):
    if k == 1:
        return False
    while n >= k:
        if n % k == 0:
            n //= k
        else:
            n %= k
    return n == 1

N = INT()

divs = divisors(N) | divisors(N-1)

ans = 0
for k in divs:
    if check(N, k):
        ans += 1
print(ans)
