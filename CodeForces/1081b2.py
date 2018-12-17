# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

# 全員同じの時、別処理
if aN.count(0):
    if aN.count(0) == N:
        print('Possible')
        print(*([1] * N))
    else:
        print('Impossible')
    exit()

for i in range(N):
    aN[i] = N - aN[i]
aN.sort()

bN = []
# bNの添字用
j = 1
# 同じ帽子の人数カウント
cnt = 0
for i in range(N):
    cnt += 1
    if aN[i] == cnt:
        bN.extend([j] * cnt)
        j += 1
        cnt = 0

if len(bN) == N: 
    print('Possible')
    print(*bN)
else:
    print('Impossible')
