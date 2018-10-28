# -*- coding: utf-8 -*-

from collections import deque

N = int(input())
aN = [int(input()) for i in range(N)]

aN_sorted = sorted(aN)
aN_sorted = deque(aN_sorted)
aN2 = [0] * N
for i in range(N):
    if i % 2 == 0:
        aN2[i] = aN_sorted.popleft()
    else:
        aN2[i] = aN_sorted.pop()
aN_sorted2 = sorted(aN, reverse=True)
aN_sorted2 = deque(aN_sorted2)
aN3 = [0] * N
for i in range(N):
    if i % 2 == 0:
        aN3[i] = aN_sorted2.popleft()
    else:
        aN3[i] = aN_sorted2.pop()

ans = 0
for i in range(1,N):
    if i == N-1:
        ans += max(abs(aN2[i-1] - aN2[i]), abs(aN2[i] - aN2[0]))
    else:
        ans += abs(aN2[i-1] - aN2[i])
ans2 = 0
for i in range(1,N):
    if i == N-1:
        ans2 += max(abs(aN3[i-1] - aN3[i]), abs(aN3[i] - aN3[0]))
    else:
        ans2 += abs(aN3[i-1] - aN3[i])

print(max(ans, ans2))