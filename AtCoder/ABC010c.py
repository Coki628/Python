# -*- coding: utf-8 -*-

"""
2点間距離でhypot使った
"""

from math import hypot

txa, tya, txb, tyb, T, V = map(int, input().split())
N = int(input())

for i in range(N):
    x, y = map(int, input().split())
    dist = hypot(txa - x, tya - y) + hypot(txb - x, tyb - y)
    if dist <= T * V:
        print('YES')
        exit()
print('NO')
