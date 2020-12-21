# -*- coding: utf-8 -*-

"""
・ビット全探索
"""

import sys
input = sys.stdin.readline

N = int(input())
aN = [int(input()) for i in range(N)]

for i in range(1 << N):
    ans = 0
    for j in range(N):
        if i >> j & 1:
            ans += aN[j]
        else:
            ans -= aN[j]
    if ans % 360 == 0:
        print('YES')
        exit()
print('NO')
