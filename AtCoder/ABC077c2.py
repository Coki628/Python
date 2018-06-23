# -*- coding: utf-8 -*-

N = int(input())
A_list = list(map(int, input().split()))
B_list = list(map(int, input().split()))
C_list = list(map(int, input().split()))

# 昇順ソート
# A_list.sort()
# C_list.sort()

ans = 0
for i in range(N):
    ab = 0
    bc = 0
    # Bを基点にしてABとBCがうまくいく数を集計していく
    for j in range(N):
        if B_list[i] > A_list[j]:
            ab += 1
        if B_list[i] < C_list[j]:
            bc += 1
    ans += ab * bc
print(ans)