# -*- coding: utf-8 -*-

N = int(input())
aN = [int(input()) for i in range(N)]

if all([a % 2 == 0 for a in aN]):
    print('second')
else:
    print('first')
