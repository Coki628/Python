# -*- coding: utf-8 -*-

"""
本番終了後、ちょっと修正版
"""

N = int(input())

if all([int(input()) % 2 == 0 for i in range(N)]):
    print('second')
else:
    print('first')
