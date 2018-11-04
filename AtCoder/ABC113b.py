# -*- coding: utf-8 -*-

N = int(input())
T,A = map(int, input().split())
hN = list(map(int, input().split()))

ans = -1
tmp = float('inf')
for i in range(N):
    t = T - hN[i] * 0.006
    if abs(t - A) < tmp:
        tmp = abs(t - A)
        ans = i + 1
print(ans)