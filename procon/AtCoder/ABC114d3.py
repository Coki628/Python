# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・ちょっと時間くったけど自力AC！
・約数の個数、素因数分解
・正しい方針は割とすんなり出たのに、全探索まとめるのに地味に時間かかった。
・最初3種類のやつしかやってなくて(しかも並びも2,4,4だけ)、何で足りないんだー？みたいな。
・まあ、当時の自分を乗り越えて自力通せたから良しとする。
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

# N!までを全部素数冪でまとめる
primes = Counter()
for i in range(2, N+1):
    primes.update(factorize(i))

# 約数が75になる素数冪の組み合わせを数える
primes = sorted(primes.values())
M = len(primes)
cnt = 0
# 3種類使う
for i, v1 in enumerate(primes):
    for j, v2 in enumerate(primes[i+1:], i+1):
        for k, v3 in enumerate(primes[j+1:], j+1):
            if v1 >= 2 and v2 >= 4 and v3 >= 4:
                cnt += 1
            if v1 >= 4 and v2 >= 2 and v3 >= 4:
                cnt += 1
            if v1 >= 4 and v2 >= 4 and v3 >= 2:
                cnt += 1
# 2種類使う
for i, v1 in enumerate(primes):
    for j, v2 in enumerate(primes[i+1:], i+1):
        if v1 >= 2 and v2 >= 24:
            cnt += 1
        if v1 >= 4 and v2 >= 14:
            cnt += 1
        if v1 >= 24 and v2 >= 2:
            cnt += 1
        if v1 >= 14 and v2 >= 4:
            cnt += 1
# 1種類使う
for i, v1 in enumerate(primes):
    if v1 >= 74:
        cnt += 1
print(cnt)
