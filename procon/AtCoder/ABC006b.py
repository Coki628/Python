# -*- coding: utf-8 -*-

N = int(input())

dp = [0] * (10 ** 6)
dp[2] = 1
for i in range(3, N):
    dp[i] = (dp[i-3] + dp[i-2] + dp[i-1]) % 10007
print(dp[N-1])
