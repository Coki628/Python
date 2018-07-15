# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

mx = 0
mn = 0
cnt = 0
if len([i for i in aN if 1 <= i < 400]) > 0:
    cnt += 1
if len([i for i in aN if 400 <= i < 800]) > 0:
    cnt += 1
if len([i for i in aN if 800 <= i < 1200]) > 0:
    cnt += 1
if len([i for i in aN if 1200 <= i < 1600]) > 0:
    cnt += 1
if len([i for i in aN if 1600 <= i < 2000]) > 0:
    cnt += 1
if len([i for i in aN if 2000 <= i < 2400]) > 0:
    cnt += 1
if len([i for i in aN if 2400 <= i < 2800]) > 0:
    cnt += 1
if len([i for i in aN if 2800 <= i < 3200]) > 0:
    cnt += 1

# 神がいる場合は話が変わってくる
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