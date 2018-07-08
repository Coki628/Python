# -*- coding: utf-8 -*-

N = int(input())

# ちょっと自分でいじった版

dp = [0] * (N+1)

for n in range(1, N+1):
    dp[n] = float("inf")
    power = 1
    while power <= n:
        dp[n] = min(dp[n], dp[n-power] + 1)
        power *= 6
    power = 1
    while power <= n:
        dp[n] = min(dp[n], dp[n-power] + 1)
        power *= 9

print(dp[N])