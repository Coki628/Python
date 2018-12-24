# -*- coding: utf-8 -*-

"""
参考：http://garnacha.techblog.jp/archives/38634648.html
　　　https://suikaba.hatenablog.com/entry/2017/07/02/234925
・二次元DP、配るDP
・jの添字に何を当てるかとか、自力で作るにはまだ練習が必要。
"""

N = int(input())
aN = list(map(int, input().split()))

# i問目までの問題を使って、j点の合計点ができるかどうかの真理値
dp = [[False] * (N*100+1) for i in range(N+1)]
dp[0][0] = True

for i in range(N):
    for j in range(N*100+1):
        if dp[i][j]:
            # 前回iでできた点数jはもちろんできる
            dp[i+1][j] = dp[i][j]
            # 前回iでできた点数j+今回追加分aN[i]もできる
            dp[i+1][j+aN[i]] = dp[i][j]

print(dp[N].count(True))
