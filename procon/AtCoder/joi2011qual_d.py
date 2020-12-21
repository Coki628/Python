# -*- coding: utf-8 -*-

"""
・さくっと自力AC
・1次元に2つ条件つけた3次元DP
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
MOD = 10000

N, K = MAP()
A = [-1] * N
for _ in range(K):
    a, b = MAP()
    a -= 1; b -= 1
    A[a] = b

# dp[i][j][k] := i日目まで見て、前々日がj、前日がkだった状態の通り数
dp = list3d(N+1, 3, 3, 0)
dp[0][0][0] = 1
for i, a in enumerate(A):
    for j in range(3):
        for k in range(3):
            for l in range(3):
                # パスタが確定している日なら、それと違うものはダメ
                if a != -1 and a != l:
                    continue
                # 3日連続同じものはダメ(最初の2日は除く)
                if i >= 2 and j == k == l:
                    continue
                dp[i+1][k][l] += dp[i][j][k]
                dp[i+1][k][l] %= MOD
ans = 0
for j in range(3):
    for k in range(3):
        ans += dp[N][j][k]
        ans %= MOD
print(ans)
