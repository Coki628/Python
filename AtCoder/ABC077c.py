# -*- coding: utf-8 -*-

N = int(input())
A_list = list(map(int, input().split()))
B_list = list(map(int, input().split()))
C_list = list(map(int, input().split()))

# 昇順ソート
A_list.sort()
B_list.sort()
C_list.sort()

ans = 0
for i in range(N):
    for j in range(N):
        # 下段 <= 中段 になったらもうやらない
        if C_list[i] <= B_list[j]:
            break
        for k in range(N):
            # 中段 <= 下段 になったらもうやらない
            if B_list[j] <= A_list[k]:
                break
            ans += 1
print(ans)