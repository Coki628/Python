# -*- coding: utf-8 -*-

"""
・ナップザック系数え上げDP
・内容を理解できるようになったので、コメントを整理して再提出
"""

N, M, K = map(int, input().split())
MOD = 998244353

# dp[i][j] := i個目のレンガまで考えた時、色の境界がj箇所ある時の通り数
dp = [[0] * 2002 for i in range(2002)]
# 初期化：最初の1つはM色好きな色で塗れる
dp [1][0] = M

for i in range(1, N):
    for j in range(K+1):
        # 次のブロックを同じ色で塗る時、通り数は増えない
        dp[i+1][j] += dp[i][j]
        dp[i+1][j] %= MOD
        # 違う色で塗る時、現在の通り数*今の色以外(M-1)が増える
        dp[i+1][j+1] += (dp[i][j] * (M-1))
        dp[i+1][j+1] %= MOD

print(dp[N][K])
