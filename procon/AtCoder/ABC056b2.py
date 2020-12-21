# -*- coding: utf-8 -*-

W, a, b = map(int, input().split())

if abs(a - b) > W:
    print(abs(a - b) - W)
else:
    print(0)