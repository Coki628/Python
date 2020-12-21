# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

mx = 0
mn = 0
cnt = 0

a = [0] * 8
for i in range(N):
    if 1 <= aN[i] < 400:
        a[0] = 1
    elif 400 <= aN[i] < 800:
        a[1] = 1
    elif 800 <= aN[i] < 1200:
        a[2] = 1
    elif 1200 <= aN[i] < 1600:
        a[3] = 1
    elif 1600 <= aN[i] < 2000:
        a[4] = 1
    elif 2000 <= aN[i] < 2400:
        a[5] = 1
    elif 2400 <= aN[i] < 2800:
        a[6] = 1
    elif 2800 <= aN[i] < 3200:
        a[7] = 1
cnt = sum(a)

kami = len([i for i in aN if 3200 <= i])
if kami == 0:
    print(str(cnt) + " " + str(cnt)) 
else:
    if cnt == 0:
        mn = 1
        mx = kami
    else:
        mn = cnt
        mx = cnt + kami
    print(str(mn) + " " + str(mx))