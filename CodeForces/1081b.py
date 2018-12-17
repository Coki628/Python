# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

aN2 = [0] * N
for i in range(N):
    aN2[i] = N - aN[i]
aN2.sort()

bN = []
# bNの添字用
j = 1
# 同じ帽子の人数カウント
cnt = 0
for i in range(N):
    cnt += 1
    if aN2[i] == cnt:
        bN.extend([j] * cnt)
        j += 1
        cnt = 0

if len(bN) == N: 
    print('Possible')
    print(*bN)
else:
    print('Impossible')
