# -*- coding: utf-8 -*-

"""
参考：http://garnacha.techblog.jp/archives/39222961.html
　　　https://blog.rossywhite.com/2018/07/30/typical-dp-contest-b/
　　　http://www.creativ.xyz/tdpc-b-684
・ゲームDP、貰うDP
・ゴールから逆算
・片方のプレイヤー側から見た場合の状態を考える
"""

A, B = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# 残りの山の数が(i, j)の時に、先手番がこれ以降にとる価値
dp = [[0] * (B+1) for i in range(A+1)]
dp[0][0] = 0

for i in range(A+1):
    for j in range(B+1):
        # 先手番かどうか
        if (A + B - i - j) % 2 == 0:
            # 両方の山に物がある(DP表の上から取るか左から取るか)
            if i != 0 and j != 0:
                dp[i][j] = max(dp[i-1][j] + a[A-i], dp[i][j-1] + b[B-j])
            # 左の山にはある(DP表の上から取る)
            elif i != 0:
                dp[i][j] = dp[i-1][j] + a[A-i]
            # 右の山にはある(DP表の左から取る)
            elif j != 0:
                dp[i][j] = dp[i][j-1] + b[B-j]
        # 後手番は得点が悪くなるように行動する
        # 先手番ではないので山a,bからの得点追加はない
        else:
            if i != 0 and j != 0:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1])
            elif i != 0:
                dp[i][j] = dp[i-1][j]
            elif j != 0:
                dp[i][j] = dp[i][j-1]

print(dp[A][B])
