# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

cnt = 0
for i in range(N):
    while aN[i] % 2 == 0 or aN[i] % 3 == 2:
        aN[i] -= 1
        cnt += 1
print(cnt)
