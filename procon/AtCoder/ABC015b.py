# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

cnt, sm = 0, 0
for i in range(N):
    if aN[i] != 0:
        cnt += 1
        sm += aN[i]
print(-(-sm // cnt))
