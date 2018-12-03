# -*- coding: utf-8 -*-

N, W = map(int, input().split())

val_wei = [list(map(int, input().split())) for i in range(N)]

dp = [[0] * (W+1) for i in range(N+1)]

for n in range(N):
    for w in range(W+1):
        if w >= val_wei[n][1]:
            dp[n+1][w] = max(dp[n][w-val_wei[n][1]] + val_wei[n][0], dp[n][w])
        else:
            dp[n+1][w] = dp[n][w]

print(dp[N][W])