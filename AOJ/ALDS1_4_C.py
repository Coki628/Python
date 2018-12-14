# -*- coding: utf-8 -*-

"""
辞書
"""

d = {}
N = int(input())
for i in range(N):
    com, s = input().split()
    if com == 'insert':
        d[s] = True
    elif com == 'find':
        if d.get(s):
            print('yes')
        else:
            print('no')
