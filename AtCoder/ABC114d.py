# -*- coding: utf-8 -*-

"""
色々自作した
"""

from math import factorial, sqrt

# 約数の個数
def num_div(num):
    total = 1
    # 終点はルート切り捨て+1
    end = int(sqrt(num)) + 1
    for i in range(2, end+1):
        cnt = 0
        # 素因数分解：小さい方から割れるだけ割って指数をカウント
        while num % i == 0:
            num //= i
            cnt += 1
        # 指数+1をかけていくと約数をカウントできる
        total *= (cnt + 1)
        # 最後までそのまま来たやつは素数なので2つ
        if i == end and total == 1:
            return 2
    return total

# 約数の列挙
def num_div_set(N):
    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て+1
    end = int(sqrt(N)) + 1
    for i in range(2, end+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N = int(input())

num = factorial(N)

cnt = 0
for i in num_div_set(num):
        if num_div(i) == 75:
            cnt += 1
print(cnt)
