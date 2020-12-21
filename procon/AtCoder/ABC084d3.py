# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
・素数列挙、累積和
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

def eratosthenes_sieve(n):
    """ 素数列挙(エラトステネスの篩) """

    table = [0] * (n + 1)
    prime_list = []
    for i in range(2, n + 1):
        if table[i] == 0:
            prime_list.append(i)
            for j in range(i + i, n + 1, i):
                table[j] = 1
    return prime_list

MAX = 10 ** 5
# 素数列挙
primes = set(eratosthenes_sieve(MAX))
acc = [0] * (MAX+1)
for a in range(3, MAX+1, 2):
    # 条件を満たす数を確認
    if a in primes and (a + 1) // 2 in primes:
        acc[a] = 1
# 累積和
acc = list(accumulate(acc))

for i in range(INT()):
    l, r = MAP()
    # 各クエリO(1)で対応
    ans = acc[r] - acc[l-1]
    print(ans)
