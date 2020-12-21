# -*- coding: utf-8 -*-

N = int(input())

if N % 2 == 0:
    if N % 4 == 0:
        print(0)
    else:
        print(1)
else:
    if (N-1) % 4 == 0:
        print(1)
    else:
        print(0)
