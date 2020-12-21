# -*- coding: utf-8 -*-

"""
・なんとか自力AC！さすがに春合宿は手強い。
・素因数分解応用
・内側while2つの3重ループがTLEしないか不安だったけど、
　素因数の数はそんなに大きくならないから大丈夫みたい。
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

def factorize(num: int) -> dict:
    """ 素因数分解 """
    from math import sqrt
    from collections import Counter

    d = Counter()
    # 終点はルート切り捨て+1
    for i in range(2, int(sqrt(num))+1):
        # 素因数分解：小さい方から割れるだけ割って素数をカウント
        while num % i == 0:
            num //= i
            d[i] += 1
        # 1まで分解したら終了
        if num == 1:
            break
    # 最後に残ったnumは素数(ただし1^1は1^0なので数に入れない)
    if num != 1:
        d[num] += 1
    return d

N = INT()

C = factorize(N)
mx = 0
for k, v in C.items():
    # どこまで階乗すれば素因数kがv個以上になるかを調べる
    sm = 0
    cur = k
    while 1:
        i = 0
        # kの倍数を1つずつ見ていって、素因数kが含まれる数だけ足していく
        while cur % (k**(i+1)) == 0:
            i += 1
        sm += i
        if sm >= v:
            break
        cur += k
    # 各素因数の中で一番数が大きくなった場所が答え
    mx = max(mx, cur)
print(mx)
