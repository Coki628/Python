# -*- coding: utf-8 -*-

import math

N = int(input())
rN = [int(input()) for i in range(N)]

rN.sort(reverse=True)
# 最後の円を見る用
rN.append(0)

ans = 0
for i in range(N):
    tmp = (rN[i] ** 2 - rN[i+1] ** 2) * math.pi
    # iが偶数番目の時は赤
    if i % 2 == 0:
        ans += tmp
print(ans)