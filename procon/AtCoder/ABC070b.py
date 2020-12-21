# -*- coding: utf-8 -*-

A, B, C, D = map(int, input().split())

ans = 0
for i in range(0, 100):
    if A <= i < B and C <= i < D:
        ans += 1
print(ans)