# -*- coding: utf-8 -*-

N = int(input())
wN = [input() for i in range(N)]

wN2 = list(set(wN))

if len(wN) != len(wN2):
    print('No')
    exit()

for i in range(1, N):
    if wN[i][0] != wN[i-1][-1]:
        print('No')
        exit()
print('Yes')