# -*- coding: utf-8 -*-

A = int(input())

ans = 0
for i in range(1, A//2+1):
    ans = max(i * (A - i), ans)
print(ans)