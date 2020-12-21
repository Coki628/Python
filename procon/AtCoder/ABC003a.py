# -*- coding: utf-8 -*-

N = int(input())

ans = 0
for i in range(N):
    ans += i+1
print(int(ans / N * 10000))
