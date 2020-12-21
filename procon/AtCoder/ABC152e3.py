# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc152/editorial.pdf
・公式解の方針もやっておく。
・素因数分解とGCD,LCM
・LCMがでかくなりすぎるので、素数冪の状態で管理する。
・素数冪のdictにA|Bでいけるんだけど、何故か遅いので普通にループ回す。
・pypyAC0.4秒、pythonもAC1.2秒。
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

def fermat(x, y, MOD): return x * pow(y, MOD-2, MOD) % MOD

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

def mul(C1, C2):
    return C1 + C2

def div(C1, C2):
    return C1 - C2

def gcd(C1, C2):
    for k, v in C2.items():
        C1[k] = min(C1[k], v)
    return C1
    # return C1 & C2

def lcm(C1, C2):
    for k, v in C2.items():
        C1[k] = max(C1[k], v)
    return C1
    # これでもできるけど遅い
    # return C1 | C2

N = INT()
A = LIST()

# 素数冪の状態で全体のLCMを出す
A2 = [factorize(a) for a in A]
l2 = Counter()
for a in A2:
    l2 = lcm(l2, a)

# 通常の値に戻す
l = 1
for k, v in l2.items():
    l *= pow(k, v, MOD)
    l %= MOD

ans = 0
for a in A:
    ans += fermat(l, a, MOD)
    ans %= MOD
print(ans)
