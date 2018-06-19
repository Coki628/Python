# -*- coding: utf-8 -*-

N = int(input())

L0 = 2
L1 = 1

if N == 1:
    print(1)
    exit()

for i in range(2, 87):

    Li = L1 + L0
    if N == i:
        print(Li)
        exit()
    else:
        L0 = L1
        L1 = Li