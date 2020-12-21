# -*- coding: utf-8 -*-

W, a, b = map(int, input().split())

if a < b:
    if b - (a + W) > 0:
        print(b - (a + W))
    else:
        print(0)

elif a > b:
    if a - (b + W) > 0:
        print(a - (b + W))
    else:
        print(0)

elif a == b:
    print(0)