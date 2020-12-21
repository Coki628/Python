# -*- coding: utf-8 -*-

a = int(input())
b = int(input())

print(min(abs(a - b), abs(abs(a - b) - 10)))
