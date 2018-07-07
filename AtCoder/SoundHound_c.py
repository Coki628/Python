# -*- coding: utf-8 -*-

import itertools
 
n, m, d = map(int, input().split())

# 重複順列で全パターン出す
nm = list(itertools.product([i for i in range(1, n+1)], repeat=m))

avg = []
for tmp in nm:
    bty = 0
    for i in range(m-1):
        if abs(tmp[i] - tmp[i+1]) == d:
            bty += 1
    avg.append(bty)

print(sum(avg) / len(nm))