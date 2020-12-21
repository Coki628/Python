# -*- coding: utf-8 -*-

N, K = map(int, input().split())
num_list = list(map(int, input().split()))

# 逆順ソート
num_list.sort(reverse=True)
# K番目まで足し合わせる
ans = 0
for i in range(K):
    ans += num_list[i]
print(ans)