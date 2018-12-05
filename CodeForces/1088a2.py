# -*- coding: utf-8 -*-

"""
解説より。O(1)でやる計算。
"""

x = int(input())

if x == 1:
    print(-1)
else:
    print(x - x % 2, 2)
