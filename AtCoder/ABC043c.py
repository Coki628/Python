# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

ans = float('inf')
for i in range(min(aN), max(aN)+1):
    sum1 = 0
    for j in range(N):
        sum1 += (aN[j] - i) ** 2
    ans = min(sum1, ans)
print(ans)