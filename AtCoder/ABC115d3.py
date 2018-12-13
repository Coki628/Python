# -*- coding: utf-8 -*-

"""
ひとつ前だけ取っておく
"""

N, X = map(int, input().split())

cur = prev = 'P'
for i in range(1, N+1):
    cur = prev = 'B' + prev + 'P' + prev + 'B'

print(cur[:X].count('P'))
