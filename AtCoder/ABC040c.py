# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

dp = [0] * N

# 初期値設定(1つめと2つめ)
dp[0] = 0
dp[1] = abs(aN[1] - aN[0])

for i in range(2,N):
    # 1つ前から来た時と2つ前から来た時の小さい方を取る
    dp[i] = min(dp[i-1] + abs(aN[i] - aN[i-1]), dp[i-2] + abs(aN[i] - aN[i-2]))

print(dp[N-1])