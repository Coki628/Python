# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC
・1次元DP、遷移も3種類あるもののやってることは素直。
・ちょっと速度が気になったので、毎回j乗してたのを毎回*jするように修正。
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

# dp[i] := i円引き出すのに必要な最小回数
dp = [INF] * (N+1)
dp[0] = 0
for i in range(N):
    # 3種類の引き出し方に対応する遷移をする
    dp[i+1] = min(dp[i+1], dp[i] + 1)
    j = 6
    while i+j <= N:
        dp[i+j] = min(dp[i+j], dp[i] + 1)
        j *= 6
    j = 9
    while i+j <= N:
        dp[i+j] = min(dp[i+j], dp[i] + 1)
        j *= 9
print(dp[N])
