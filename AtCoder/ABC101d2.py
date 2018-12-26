# -*- coding: utf-8 -*-

K = int(input())

if K <= 0:
    exit()

out = 0
i = 1
while True:

    if 1 <= i <= 9 or (i >= 10 and i % 10 == 9): 
        if i >= 1000000000000000:
            exit()
        print(i)
        out += 1
        if out == K:
            exit()
    i += 1
