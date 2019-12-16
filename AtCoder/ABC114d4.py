# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/abc114/submissions/8965547
・さすがに全探索が不格好だから綺麗にまとまらないかなーと思って人の参考に。
・確かにこうやって予め組み合わせに使える個数を数えておくとうまくまとまるなー。
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
cnt2 = cnt4 = cnt14 = cnt24 = cnt74 = 0
for a in primes:
    # 組み合わせに使える数を予め数えておく
    if a >= 2: cnt2 += 1
    if a >= 4: cnt4 += 1
    if a >= 14: cnt14 += 1
    if a >= 24: cnt24 += 1
    if a >= 74: cnt74 += 1

ans = 0
# 3種類使う
ans += (cnt2-2) * (cnt4 * (cnt4-1) // 2)
# 2種類使う
ans += (cnt2-1) * cnt24 + (cnt4-1) * cnt14
# 1種類使う
ans += cnt74

print(ans)
