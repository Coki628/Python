# -*- coding: utf-8 -*-

from itertools import combinations

aN = list(map(int, input().split()))

comb = list(combinations(aN, 3))

sums = [0] * len(comb)
for i in range(len(comb)):
    sums[i] = sum(comb[i])

sums.sort(reverse=True)

print(sums[2])