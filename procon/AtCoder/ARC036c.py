# -*- coding: utf-8 -*-

"""
・自力WA
・桁DPで数え上げ、っていう方針自体は良かったけど、
　そこまでの連続する最大数をどう持てばいいかが掴めなかった。
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

N, K = MAP()
S = input()

dp = list3d(N+1, 2, K+1, 0)
dp[0][0][0] = 1
for i in range(N):
    s = S[i]
    for j in range(2):
        for k in range(K+1):
            if s == '?':
                if j == 0:
                    dp[i+1][1][1] += dp[i][j][k]
                    if k+1 < K:
                        dp[i+1][0][k+1] += dp[i][j][k]
                else:
                    dp[i+1][0][1] += dp[i][j][k]
                    if k+1 < K:
                        dp[i+1][1][k+1] += dp[i][j][k]
            elif s == '0':
                if j == 0:
                    if k+1 < K:
                        dp[i+1][0][k+1] += dp[i][j][k]
                else:
                    dp[i+1][0][1] += dp[i][j][k]
            else:
                if j == 0:
                    dp[i+1][1][1] += dp[i][j][k]
                else:
                    if k+1 < K:
                        dp[i+1][1][k+1] += dp[i][j][k]
ans = 0
for j in range(2):
    for k in range(K+1):
        ans += dp[N][j][k]
print(ans)
