# -*- coding: utf-8 -*-

"""
・ゲームDP(非再帰)、区間DP
・ABC163Eで2乗の区間DPをやったので、同じ要領でこれ出来ないかと思って。
・後ろからやるから逆向きではあるけど、結果的に遷移のロジックとしてはほぼ同じものを適用できた。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

# dp[i][j] := 左からi個、右からj個取った状態での最大(最小)得点
dp = list2d(N+1, N+1, 0)
# 初期化：どちらの手番で使う場所かに合わせて-INFかINFにする
for i in range(N):
    turn = i & 1
    for j in range(i+1):
        k = i - j
        if turn == 0:
            dp[j][k] = -INF
        else:
            dp[j][k] = INF
# 最初の遷移の開始地点(ゲーム終了の状態)は0に
for i in range(N+1):
    dp[i][N-i] = 0
# 終了から順番に戻って見ていく
for i in range(N-1, -1, -1):
    turn = i & 1
    for j in range(i+1):
        k = i - j
        l = j
        r = N - 1 - k
        if turn == 0:
            # 先手番の遷移
            dp[j][k] = max(dp[j][k], dp[j+1][k] + A[l])
            dp[j][k] = max(dp[j][k], dp[j][k+1] + A[r])
        else:
            # 後手番の遷移
            dp[j][k] = min(dp[j][k], dp[j+1][k] - A[l])
            dp[j][k] = min(dp[j][k], dp[j][k+1] - A[r])
ans = dp[0][0]
print(ans)
