# -*- coding: utf-8 -*-

from math import factorial

mod = 1000000007
W,H = map(int, input().split())
n = W + H - 2
r = W - 1

# 組み合わせ(comb)の総数
print((factorial(n) // (factorial(n - r) * factorial(r))) % mod)