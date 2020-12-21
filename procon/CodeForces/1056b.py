# -*- coding: utf-8 -*-

n,m = map(int, input().split())

cnt = 0
for i in range(1, n+1):
    for j in range(1, n+1):
        if (i ** 2 + j ** 2) % m == 0:
            cnt += 1
print(cnt)
