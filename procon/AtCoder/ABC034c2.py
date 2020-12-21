# -*- coding: utf-8 -*-

from math import factorial

mod = 10 ** 9 + 7
W,H = map(int, input().split())
n = W + H - 2
r = W - 1

# ここからmodの世界(合同式)
n = n % mod
r = r % mod
# 割り算前に、分子と分母を整理
numer = factorial(n) % mod
denomin = (factorial(n - r) % mod) * (factorial(r) % mod) % mod

# modは素数なのでmod-2乗使って掛け算すれば、割り算もいける(フェルマーの小定理)
print(numer * pow(denomin, mod-2, mod) % mod)