# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

sum1 = aN[0]
cnt = 0
if aN[0] > 0:
    sei = True
elif aN[0] < 0:
    sei = False
else:
    

for i in range(1,N):
    sum1 += aN[i]
    if sei:
        if sum1 >= 0:
            tmp = abs(sum1) +1
            cnt += tmp
            sum1 -= tmp
        sei = False
    else:
        if sum1 <= 0:
            tmp = abs(sum1) +1
            cnt += tmp
            sum1 += tmp
        sei = True
print(cnt)