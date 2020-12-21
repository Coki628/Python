# -*- coding: utf-8 -*-

N = int(input())

i = 0
while i * 4 <= 100:
    j = 0
    while j * 7 <= 100:
        if i * 4 + j * 7 == N:
            print("Yes")
            exit()
        j += 1
    i += 1
print("No")