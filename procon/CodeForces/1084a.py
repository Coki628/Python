# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

ans = float('inf')
for i in range(N):
    total = 0
    for j in range(N):
        # (x階から人のいる階 + 人のいる階から1階 + 1階からx階) * 往復
        elec = (abs(i - j) + j + i) * 2
        total += aN[j] * elec
    ans = min(total, ans)
print(ans)
