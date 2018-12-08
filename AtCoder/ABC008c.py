# -*- coding: utf-8 -*-

from itertools import permutations

N = int(input())
aN = [int(input()) for i in range(N)]

perms = list(permutations(aN))
cnt = 0
for p in perms:
    # 0 = 表とする
    coins = [0] * N
    # 自分より右の確認
    for i in range(N):
        for j in range(i+1, N):
            if p[j] % p[i] == 0:
                coins[j] += 1
    # 表向きの集計
    for i in range(N):
        if coins[i] % 2 == 0:
            cnt += 1
print(cnt / len(perms))
