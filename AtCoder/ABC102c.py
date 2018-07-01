# -*- coding: utf-8 -*-

N = int(input())
AN = list(map(int, input().split()))

ans = float("inf")
for b in range(-10, 11):

    tmp = 0
    for i in range(N):
        tmp += abs(AN[i] - (b + i + 1))

    ans = min(ans, tmp)

print(ans)