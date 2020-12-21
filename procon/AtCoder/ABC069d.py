# -*- coding: utf-8 -*-

"""
・すんなり自力AC
"""

H, W = map(int, input().split())
N = int(input())
aN = list(map(int, input().split()))

# aNの中身を今回使いたい形に持ち替える
nums = []
for i in range(N):
    nums.extend([i+1] * aN[i])

# とりあえず前から順に入れる
grid = [[0] * W for _ in range(H)]
k = 0
for i in range(H):
    for j in range(W):
        grid[i][j] = nums[k]
        k += 1

# 出力の前に逆順と交互にする
for i in range(H):
    if i % 2 == 0:
        print(*grid[i])
    else:
        print(*reversed(grid[i]))
