# -*- coding: utf-8 -*-

N, K = map(int, input().split())
num_list = list(map(int, input().split()))

for i in range(1, N):

    if (K - 1) * i + 1 >= N:
        print(i)
        exit()