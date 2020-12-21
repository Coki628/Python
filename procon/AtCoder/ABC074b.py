# -*- coding: utf-8 -*-

N = int(input())
K = int(input())
num_list = list(map(int, input().split()))

ans = 0
for i in range(N):
    A = num_list[i]
    B = abs(K - A)
    if A <= B:
        ans += A * 2
    else:
        ans += B * 2
print(ans)