# -*- coding: utf-8 -*-

"""
・自力AC
・LCS(最長共通部分列)のDP
・N<=100でO(N^3)がいけるので、区切り位置全探索*LCSのDP、で大丈夫。
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

def LCS(S, T):
    N = len(S)
    M = len(T)
    dp = [[0] * (M+1) for i in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, M+1):
            # 上から受け取る
            dp[i][j] = max(dp[i][j], dp[i-1][j])
            # 左から受け取る
            dp[i][j] = max(dp[i][j], dp[i][j-1])
            # 今回増やす文字が一致するなら、LCSが1文字増える
            if S[i-1] == T[j-1]:
                # 左上から受け取る
                dp[i][j] = max(dp[i][j], dp[i-1][j-1]+1)
    return dp[N][M]

N = INT()
S = input()

ans = INF
# 前後半どこで区切るかを全探索
for i in range(N):
    s, t = S[:i], S[i:]
    lcs = LCS(s, t)
    # 消す必要のある数 = 全文字数 - 残せる文字数(LCS * 2)
    ans = min(ans, N - lcs * 2)
print(ans)
