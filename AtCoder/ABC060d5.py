# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/abc060/submissions/3958560
・前から普通にDPもdictでやってみる。
・配るDP
・2つ目の添字はdict
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from collections import defaultdict

N, W = map(int, input().split())
wN = [0] * N
vN = [0] * N
for i in range(N):
    wN[i], vN[i] = map(int, input().split())

# 重さの添字にはdictを使う
dp = [defaultdict(int) for i in range(N+1)]
dp[0][0] = 0

for i in range(N):
    for k, v in dp[i].items():
        # 重さの条件がOKなら、今回の値を足したケースを次のステップに送る
        if k + wN[i] <= W:
            dp[i+1][k+wN[i]] = max(dp[i][k] + vN[i], dp[i+1][k+wN[i]])
        # 足さないケースを送る
        dp[i+1][k] = max(dp[i][k], dp[i+1][k])

# 最適が重さWを全て使った場合とは限らないのでNステップ目の全重さパターンから価値のmaxを取る
print(max(dp[N].values()))
