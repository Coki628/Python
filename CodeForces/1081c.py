# -*- coding: utf-8 -*-

"""
解説参考：DP
・自分で思いつく気がしない…。
・図的には自分の下と右下に送る感じ
・2000 * 2000が多かったのか、pythonでTLEしてpypyで通した。
"""

N, M, K = map(int, input().split())
MOD = 998244353

dp = [[0] * 2002 for i in range(2002)]
# レンガ1つで違う色が0なら、パターン数 = 色の種類数
dp [1][0] = M

for i in range(1, N):
    for j in range(K+1):
        dp[i+1][j] = (dp[i+1][j] + dp[i][j]) % MOD
        dp[i+1][j+1] = (dp[i][j] * (M-1)) % MOD

print(dp[N][K])
