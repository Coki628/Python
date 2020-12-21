# -*- coding: utf-8 -*-

N = int(input())
sN = [int(input()) for i in range(N)]

# dpでやってみる版

# [どこまでのsNを使うか][合計として取りうる値かどうか]
# 範囲は0を含むNまでのN+1個と0～10000までの10001個
dp = [[False] * (N*100+1) for j in range(N+1)]

# 初期値設定(s1～sNを何も使わない時、合計値0は取りうるか→True)
dp[0][0] = True

for i in range(N):
    for j in range(N*100+1):
        # 前回までの設定値でここがTrueになっていたら
        if dp[i][j] == True:
            # sN合計の一つ先で、今までの値はTrue
            dp[i+1][j] = True
            # sN合計の一つ先で、今までの値+今回のsNもTrue
            dp[i+1][j+sN[i]] = True

# 高い値から順に見ていく
for i in range(N*100, -1, -1):
    if dp[N][i]:
        if i % 10 != 0:
            print(i)
            exit()
print(0)