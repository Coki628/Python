# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2018/08/05/224100_1
・文字列DP
・こういう、ちょっと状況込み入ったDPがまだ自分でうまく書けない。
・もっと数こなして経験積まなきゃなー。
"""

MOD = 10 ** 9 + 7
S = input()

# 文字列Sのi文字目まで考えたときにABCのj文字目まで照合させた個数
dp = [[0] * 4 for i in range(len(S)+1)]
dp[0][0] = 1

for i in range(len(S)):
    # i文字目でABCカウントを進めない場合(真下に送る分)
    for j in range(4):
        if S[i] == '?':
            dp[i+1][j] = (dp[i+1][j] + dp[i][j] * 3) % MOD
        else:
            dp[i+1][j] = (dp[i+1][j] + dp[i][j]) % MOD
    # i文字目でABCカウントを進める場合(右下に送る分)
    if S[i] == '?' or S[i] == 'A':
        dp[i+1][1] = (dp[i+1][1] + dp[i][0]) % MOD
    if S[i] == '?' or S[i] == 'B':
        dp[i+1][2] = (dp[i+1][2] + dp[i][1]) % MOD
    if S[i] == '?' or S[i] == 'C':
        dp[i+1][3] = (dp[i+1][3] + dp[i][2]) % MOD

print(dp[len(S)][3])
