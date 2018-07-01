# -*- coding: utf-8 -*-

W, H, N = map(int, input().split())
aN = [list(map(int, input().split())) for i in range (N)]

# 1マスずつ塗りつぶさないパターン(こっちのが速い)
minXY = [0, 0]
maxXY = [W, H]
for i in range(N):
    
    if aN[i][2] == 1:
        minXY[0] = max(minXY[0], aN[i][0])

    elif aN[i][2] == 2:
        maxXY[0] = min(maxXY[0], aN[i][0])

    elif aN[i][2] == 3:
        minXY[1] = max(minXY[1], aN[i][1])

    elif aN[i][2] == 4:
        maxXY[1] = min(maxXY[1], aN[i][1])

# 縦横それぞれにマイナスの可能性を考慮する
ansW = maxXY[0] - minXY[0]
ansH = maxXY[1] - minXY[1]
if ansW < 0:
    ansW = 0
if ansH < 0:
    ansH = 0

print(ansH * ansW)