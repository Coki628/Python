# -*- coding: utf-8 -*-

N, K = map(int, input().split())
hN = [int(input()) for i in range(N)]

hN.sort()

ans = float('inf')
for i in range(N-K+1):
    ans = min(hN[i+K-1] - hN[i], ans)
print(ans)
