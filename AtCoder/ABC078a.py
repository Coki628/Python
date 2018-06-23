# -*- coding: utf-8 -*-

X, Y = map(str, input().split())

X2 = int(X, 16)
Y2 = int(Y, 16)

if X2 == Y2:
    print("=")
elif X2 < Y2:
    print("<")
elif X2 > Y2:
    print(">")