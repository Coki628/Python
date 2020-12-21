# -*- coding: utf-8 -*-

X = int(input())

sum1 = 0
i = 1
while True:
    sum1 += i
    if sum1 >= X:
        break
    i += 1
print(i)