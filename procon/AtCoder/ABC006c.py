# -*- coding: utf-8 -*-

from itertools import product

N, M = map(int, input().split())

# 重複順列
for prod in product([2, 3, 4], repeat=N):
    if sum(prod) == M:
        print(prod.count(2), prod.count(3), prod.count(4))
        exit()
print(-1, -1, -1)
