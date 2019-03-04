# -*- coding: utf-8 -*-

C,D = map(int, input().split())

starts = []
limits = []
i = 1
while 140 * i <= 10 ** 15:
    starts.append(140 * i)
    i *= 2
i = 1
while 170 * i <= 10 ** 15:
    limits.append(170 * i)
    i *= 2

cnt = 0
for i in range(len(limits)-1):
    if limits[i] <= D < limits[i+1]:
        if D < limits[i] and C < starts[i]:
            cnt += D - limits[i]
        elif D < limits[i] and C >= starts[i]:
            cnt += D - C
        break
    elif C < starts[i]:
        cnt += limits[i] - starts[i]

print(cnt)