# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

aN.sort()
ans = 0
for i in range(0, N, 2):
    if aN[i] != aN[i+1]:
        ans += aN[i+1] - aN[i]
print(ans)
