# -*- coding: utf-8 -*-

x = int(input())

for i in range(1, x+1):
    for j in range(1, x+1):
        if (i % j == 0 and i * j > x and i / j < x):
            print(i, j)
            exit()
print(-1)
