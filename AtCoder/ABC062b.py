# -*- coding: utf-8 -*-

H, W = map(int, input().split())
HW = [list(input()) for i in range(H)]

# 入力より一回り大きい2次元配列を#埋めで準備
ans = [["#" for i in range(W+2)] for j in range(H+2)]
# 必要な部分を入力の配列で上書き
for i in range(H):
    for j in range(W):
        ans[i+1][j+1] = HW[i][j]

for i in range(H+2):
    print("".join(ans[i]))