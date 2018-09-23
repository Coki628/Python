# -*- coding: utf-8 -*-

A, B, C = map(str, input().split())

print(max(eval(A + '+' + B + C), eval(A + B + '+' + C), eval(B + '+' + C + A),
        eval(A + '+' + C + B), eval(B + A + '+' + C), eval(B + '+' + A + C)))