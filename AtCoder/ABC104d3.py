# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力ならず。。これは解きたかったな。
・文字列DP
・DP考えたけど、状態をうまく作れなかった。
・j文字選んだ状態、を0〜3の4つ持っておくとうまくいく。
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

S = input()
N = len(S)

# dp[i][j] := i文字目まで見て、ABCのうちj文字目まで選んだ状態の通り数
dp = list2d(N+1, 4, 0)
dp[0][0] = 1
for i in range(N):
    s = S[i]
    for j in range(4):
        # 今回の文字を選ばない
        if s == '?':
            # ?は3通り遷移させる
            dp[i+1][j] += dp[i][j] * 3
            dp[i+1][j] %= MOD
        else:
            dp[i+1][j] += dp[i][j]
            dp[i+1][j] %= MOD
        # 今回の文字を選ぶ
        if j < 3:
            # 条件に合わせて次の文字へ進む
            if s == '?' or s == 'A' and j == 0 \
                        or s == 'B' and j == 1 \
                        or s == 'C' and j == 2:
                dp[i+1][j+1] += dp[i][j]
                dp[i+1][j+1] %= MOD
# N文字目まで見て、3文字選び終わっている通り数
print(dp[N][3])
