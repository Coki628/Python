# -*- coding: utf-8 -*-

N,T = map(int, input().split())

cNtN = [[0] * 2 for i in range(N)]
for i in range(N):
    cNtN[i][0], cNtN[i][1] = map(int, input().split())

for i in range(N):
    if cNtN[i][1] > T:
        cNtN[i][0] = float('inf')

min_c = float('inf')
for i in range(len(cNtN)):
    min_c = min(min_c, cNtN[i][0])

if min_c == float('inf'):
    print('TLE')
else:
    print(min_c)