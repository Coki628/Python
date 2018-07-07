# -*- coding: utf-8 -*-

N = int(input())

# 解説動画通りにDPしてみる

dp = [0] * 100001
# dp[0] = 0

for n in range(1, 100001):
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