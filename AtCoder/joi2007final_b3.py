# -*- coding: utf-8 -*-

"""
・連続する部分列のLCSっぽいDP
・ABC141Eからほぼ借用。pypyなら速度は間に合うようになったけどMLE。。
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
T = input()
M = len(T)

# dp[i][j] := iからとjから始めた時に、一致する最大の長さ
dp = list2d(N+1, M+1, 0)
# 後ろから埋めていく
for i in range(N-1, -1, -1):
    for j in range(M-1, -1, -1):
        if S[i] == T[j]:
            dp[i][j] = dp[i+1][j+1] + 1

ans = 0
for i in range(N+1):
    for j in range(M+1):
        ans = max(ans, dp[i][j])
print(ans)
