# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

for i in range(N):
    while aN[i] % 2 == 0:
        aN[i] //= 2

print(len(set(aN)))
