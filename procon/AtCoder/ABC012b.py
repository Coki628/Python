# -*- coding: utf-8 -*-

N = int(input())

h = N // 3600
N %= 3600
m = N // 60
s = N % 60

print(str(h).zfill(2) + ':' + str(m).zfill(2) + ':' + str(s).zfill(2))
