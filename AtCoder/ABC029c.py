# -*- coding: utf-8 -*-

from itertools import product

N = int(input())

# 重複順列(辞書順は多分もうできてる)
p = list(product(['a','b','c'], repeat=N))

for i in range(len(p)):
    print(''.join(p[i]))