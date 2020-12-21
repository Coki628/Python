# -*- coding: utf-8 -*-

a, b, c = map(int, input().split())

if a + b == c or a == b + c or a + c == b:
    print("Yes")
else:
    print("No")