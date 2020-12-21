# -*- coding: utf-8 -*-

import sys
def input(): return sys.stdin.readline().strip()

N, K = map(int, input().split())
xN, yN, cN = [0]*N, [0]*N, ['']*N
B = [[0] * (K*2) for i in range(K*2)]
W = [[0] * (K*2) for i in range(K*2)]
cnt = 0
for i in range(N):
    x, y, c = input().split()
    xN[i], yN[i], cN[i] = int(x) % (K*2), int(y) % (K*2), c
    if cN[i] == 'W':
        W[yN[i]][xN[i]] += 1
        if (xN[i] < K and yN[i] < K
                or xN[i] >= K and yN[i] >= K):
            cnt += 1
    if cN[i] == 'B':
        B[yN[i]][xN[i]] += 1
        if (xN[i] < K and yN[i] >= K
                or xN[i] >= K and yN[i] < K):
            cnt += 1
