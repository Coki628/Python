# -*- coding: utf-8 -*-

W, H, N = map(int, input().split())
HW = [[0 for i in range(W)] for j in range(H)]

aN = [list(map(int, input().split())) for i in range (N)]

for i in range(N):

    if aN[i][2] == 1:
        for j in range(H):
            for k in range(aN[i][0]):
                HW[j][k] = 1

    elif aN[i][2] == 2:
        for j in range(H):
            for k in range(aN[i][0], W):
                HW[j][k] = 1

    elif aN[i][2] == 3:
        for j in range(aN[i][1]):
            for k in range(W):
                HW[j][k] = 1

    elif aN[i][2] == 4:
        for j in range(aN[i][1], H):
            for k in range(W):
                HW[j][k] = 1

ans = 0
for i in range(H):
    ans += HW[i].count(0)
print(ans)