# -*- coding: utf-8 -*-

K = int(input())

if K % 2 == 0:
    print((K // 2) ** 2)
else:
    print((K//2 + 1) * (K//2))