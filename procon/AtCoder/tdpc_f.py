# -*- coding: utf-8 -*-

"""
参考：https://suikaba.hatenablog.com/entry/2017/07/03/130702
・自力ならず。愚直なやつしか思いつかなかった。。
・どの駅にも移動できるから、どうやってまとめて遷移させよう、とか考えてしまったけど、
　停車or通過(j=0,1)、の2択を毎回やってると考えると添字2が2択だけでいいのは気付きたかった。
・とりあえず全部遷移させて、その通り数から余計な遷移部分を引けばいいって発想は新しかった。
・N<=100万でpypyAC0.6秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, K = MAP()

# dp[i][j] := 駅iまで考えた時、その駅で停車or通過(j=0,1)する通り数
dp = list2d(N+1, 2, 0)
# 駅0を通過中は番兵的に1にしておく
dp[0][1] = dp[1][0] = 1
for i in range(2, N+1):
    for j in range(2):
        for k in range(2):
            dp[i][j] += dp[i-1][k]
            dp[i][j] %= MOD
    # K-1個前の駅からここまで連続で停車できる通り数 = K個前には通過中の通り数、を引く
    if i-K >= 0:
        dp[i][0] -= dp[i-K][1]
        dp[i][0] %= MOD
# 最後の駅で停車している通り数
print(dp[N][0])
