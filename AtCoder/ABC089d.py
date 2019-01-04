# -*- coding: utf-8 -*-

"""
自力AC！しかも結構悩まないでいけた！
"""

import sys
input = sys.stdin.readline

H, W, D = map(int, input().split())
HW = []
for i in range(H):
    HW.append(list(map(int, input().split())))

# 各マス番号の座標位置をメモしておく
pos = [None] * (H*W+1)
for i in range(H):
    for j in range(W):
        pos[HW[i][j]] = (i, j)
# 間隔をDずつ空けた累積和
sm = [0] * (H*W+1)
for i in range(D+1, H*W+1):
    sm[i] = sm[i-D] + abs(pos[i][0] - pos[i-D][0]) + abs(pos[i][1] - pos[i-D][1])
# 各QはO(1)で解けるはず
Q = int(input())
for i in range(Q):
    l, r = map(int, input().split())
    # 右端 - 左端で特定区間の合計も出せる
    print(sm[r] - sm[l])
