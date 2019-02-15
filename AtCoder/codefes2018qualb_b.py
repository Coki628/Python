# -*- coding: utf-8 -*-

N,X = map(int, input().split())
aNbN = [[0] * 2 for i in range(N)]
for i in range(N):
    aNbN[i][0], aNbN[i][1] = map(int, input().split())

max_idx = 0
max_b = 0
for i in range(N):
    if aNbN[i][1] >= max_b:
        max_b = aNbN[i][1]
        max_idx = i

aNbN[max_idx][0] += X

ans = 0
for i in range(N):
    ans += aNbN[i][0] * aNbN[i][1]

print(ans)