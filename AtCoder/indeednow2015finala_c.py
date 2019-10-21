# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-2
・LCS系2方向DPの発展、3方向DP
・分かってしまえばやることはシンプルだけど、問題を見てこの方針を導けるかどうか。
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

N, M = MAP()
# dp[i][j][k] := 3つの能力がそれぞれi,j,kの時に得られる最大値
dp = list3d(101, 101, 101, 0)
for i in range(N):
    a, b, c, w = MAP()
    # 会社側の条件を入れる
    dp[a][b][c] = max(dp[a][b][c], w)

for i in range(101):
    for j in range(101):
        for k in range(101):
            # 3方向から貰うDP
            if i != 0:
                dp[i][j][k] = max(dp[i][j][k], dp[i-1][j][k])
            if j != 0:
                dp[i][j][k] = max(dp[i][j][k], dp[i][j-1][k])
            if k != 0:
                dp[i][j][k] = max(dp[i][j][k], dp[i][j][k-1])

for i in range(M):
    a, b, c = MAP()
    # 応募者毎にO(1)で取得可能
    print(dp[a][b][c])
