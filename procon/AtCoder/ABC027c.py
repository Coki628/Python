# -*- coding: utf-8 -*-

N = int(input())

cnt = 0
x = N
while x != 1:
    if x % 2:
        x = (x - 1) // 2
    else:
        x //= 2
    cnt += 1

if cnt % 2:
    print('Takahashi')
else:
    print('Aoki')