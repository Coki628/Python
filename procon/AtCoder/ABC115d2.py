# -*- coding: utf-8 -*-

"""
DP
"""

N, X = map(int, input().split())

dp =[''] * (N+1)
dp[0] = 'P'

for i in range(1, N+1):
    dp[i] = 'B' + dp[i-1] + 'P' + dp[i-1] + 'B'

print(dp[N][:X].count('P'))
