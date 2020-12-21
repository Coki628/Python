# -*- coding: utf-8 -*-

N = int(input())
sN = [int(input()) for i in range(N)]

sN.sort()
sumS = sum(sN)

if sumS % 10 != 0:
    print(sumS)
    exit()
else:
    for i in range(N):
        if sN[i] % 10 != 0:
            sumS -= sN[i]
            print(sumS)
            exit()
    print(0)