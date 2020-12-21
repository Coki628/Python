# -*- coding: utf-8 -*-

"""
本番終了後、ちょっと修正版
"""

from collections import defaultdict
from math import sqrt

# 素因数分解
def fact_prime(num):
    d = defaultdict(int)
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end+1):
        cnt = 0
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

N, P = map(int, input().split())

d = fact_prime(P)
ans = 1
for k, v in d.items():
    ans *= k ** (v // N)
print(ans)
