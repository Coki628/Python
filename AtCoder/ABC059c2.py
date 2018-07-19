# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

sum1 = 0
cnt1 = 0
for i in range(N):
    sum1 += aN[i]
    if i % 2 == 0:
        if sum1 >= 0:
            cnt1 += abs(sum1) +1
            sum1 -= abs(sum1) +1
    else:
        if sum1 <= 0:
            cnt1 += abs(sum1) +1
            sum1 += abs(sum1) +1

sum1 = 0
cnt2 = 0
for i in range(N):
    sum1 += aN[i]
    if i % 2 == 1:
        if sum1 >= 0:
            cnt2 += abs(sum1) +1
            sum1 -= abs(sum1) +1
    else:
        if sum1 <= 0:
            cnt2 += abs(sum1) +1
            sum1 += abs(sum1) +1

# 開始を偶奇それぞれでやって、良かった方を出力
print(min(cnt1, cnt2))