# -*- coding: utf-8 -*-

X, Y = map(int, input().split())

if X <= 1 and Y <= 1 or X == 2 and Y == 1 or X == 1 and Y == 2 or X == 3 and Y == 1 or X == 1 and Y == 3:
    print('Brown')
else:
    print('Alice')
