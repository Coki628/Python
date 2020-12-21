# -*- coding: utf-8 -*-

""" 2次元DP(マイナスの値を保持する) """

N,A = map(int, input().split())
xN = list(map(int, input().split()))
X = max(max(xN), A)
# 全てのxから-Aすることで、合計=0,平均=0を求める形にする
yN = [int(i)-A for i in xN]

# 二つ目の添え字はマイナスを取りうるので工夫する
dp = [[0] * (2*N*X+1) for j in range(N+1)]
# 基準点を0じゃなくて真ん中(N*X)にする
dp[0][N*X] = 1

ans = 0
for i in range(1,N+1):
    for t in range(2*N*X+1):
        # 範囲外ならそのまま
        if t - yN[i-1] < 0 or t - yN[i-1] > 2*N*X:
            dp[i][t] = dp[i-1][t]
        # 範囲内なら前の値を足す
        elif 0 <= t - yN[i-1] <= 2*N*X:
            dp[i][t] = dp[i-1][t] + dp[i-1][t - yN[i-1]]
# 1枚も選ばないパターンを引く
print(dp[N][N*X]-1)