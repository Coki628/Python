# -*- coding: utf-8 -*-

X, A, B = map(int, input().split())

if A - B >= 0:
    print("delicious")
elif A - B + X >= 0:
    print("safe")
else:
    print("dangerous")