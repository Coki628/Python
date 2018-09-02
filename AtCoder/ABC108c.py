# -*- coding: utf-8 -*-

import itertools

N,K = map(int, input().split())

p = list(itertools.product(range(1,N+1), repeat=3))

ans = 0
for pair in p:
    if (pair[0] + pair[1]) % K == 0 and (pair[1] + pair[2]) % K == 0 and (pair[2] + pair[0]) % K == 0:
        ans += 1

print(ans)