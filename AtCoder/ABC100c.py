# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

cnt = 0
for i in range(N):
    tmp = aN[i]
    while True:
        if tmp % 2 == 0:
            cnt += 1
            tmp //= 2
        else:
            break
print(cnt)