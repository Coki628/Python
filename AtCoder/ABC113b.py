# -*- coding: utf-8 -*-

N = int(input())
T,A = map(int, input().split())
hN = list(map(int, input().split()))

tN = [0] * N
ans = -1
tmp = float('inf')
for i in range(N):
    tN[i] = T - hN[i] * 0.006
    if abs(tN[i] - A) < tmp:
        tmp = abs(tN[i] - A)
        ans = i + 1
print(ans)