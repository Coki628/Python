# -*- coding: utf-8 -*-
 
x = int(input())
 
a = x // 11
b = x % 11
 
a *= 2
if 0 < b <= 6:
    a += 1
elif b >= 7:
    a += 2
print(a)