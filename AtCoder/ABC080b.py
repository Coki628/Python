# -*- coding: utf-8 -*-

N = int(input())

X = N
sum1 = 0
# 各桁の和を出す
while N != 0:
    sum1 += N % 10
    N //= 10
    
# 判定する
if X % sum1 == 0:
    print("Yes")
else:
    print("No")