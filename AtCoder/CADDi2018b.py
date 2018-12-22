# -*- coding: utf-8 -*-

N, H, W = map(int, input().split())
cnt = 0
for i in range(N):
    a, b = map(int, input().split())
    if a >= H and b >= W:
        cnt += 1
print(cnt)
