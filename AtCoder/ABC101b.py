# -*- coding: utf-8 -*-

N = int(input())

num = N
sum1 = 0
while num != 0:
    sum1 += num % 10
    num //= 10
if N % sum1 == 0:
    print("Yes")
else:
    print("No")