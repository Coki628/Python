# -*- coding: utf-8 -*-

N = int(input())

key = 2025 - N
for i in range(1, 10):
    for j in range(1, 10):
        if i * j == key:
            print(str(i) + ' x ' + str(j))
