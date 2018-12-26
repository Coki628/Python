# -*- coding: utf-8 -*-

K = int(input())

out = 0
i = 1
while True:

    N = i
    M = i + 1

    Sn = 0
    num = N
    while num != 0:
        Sn += num % 10
        num //= 10
    Sm = 0
    num = M
    while num != 0:
        Sm += num % 10
        num //= 10
    
    if N / Sn <= M / Sm:
        print(i)
        out += 1
        if out == K:
            exit()
    i += 1
