# -*- coding: utf-8 -*-

"""
参考：http://garnacha.techblog.jp/archives/38634648.html
　　　https://suikaba.hatenablog.com/entry/2017/07/02/234925
・二次元DP、貰うDP
・練習のため、こっちもやってみる。
・同じ点を軸にして両方の追記ができるから配る方が分かりやすいかなぁ。
　if文の場合分けもいらないし、aNとの添字も合うし。
"""

N = int(input())
aN = list(map(int, input().split()))

# i問目までの問題を使って、j点の合計点ができるかどうかの真理値
dp = [[False] * (N*100+1) for i in range(N+1)]
dp[0][0] = True

for i in range(1, N+1):
    for j in range(N*100+1):
        if dp[i-1][j]:
            # 前回i-1でできた点数j
            dp[i][j] = dp[i-1][j]
        elif j >= aN[i-1]:
            # 前回i-1でできた点数j+今回追加分aN[i-1]
            dp[i][j] = dp[i-1][j-aN[i-1]]

print(dp[N].count(True))
