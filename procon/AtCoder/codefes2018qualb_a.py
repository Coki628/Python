# -*- coding: utf-8 -*-

N = int(input())

ans = 100
for i in range(1, 101):
    if i % N == 0:
        ans -= 1
print(ans)