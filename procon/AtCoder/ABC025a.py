# -*- coding: utf-8 -*-

from itertools import product

S = list(input())
N = int(input())

# 重複順列
p = list(product(S, repeat=2))

print(''.join(p[N-1]))