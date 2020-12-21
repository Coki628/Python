# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-13、さくっと自力AC！
・グリッドDP、2種類のフラグの情報を入れる。
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
MOD = 10 ** 5

H, W = MAP()

# dp[0/1][0/1][i][j] := 直前に曲がったかどうか(0/1)、直前は上か右か(0/1)、の状態でグリッド位置(i, j)に到達する通り数
dp = list4d(2, 2, H+1, W+1, 0)
# 初期化：(0, 0)から上に行く場合と右に行く場合
dp[0][0][1][0] = 1
dp[0][1][0][1] = 1

for i in range(H):
    for j in range(W):
        # フラグなし↑↑
        dp[0][0][i+1][j] += dp[0][0][i][j]
        # フラグなし↑→
        dp[1][1][i][j+1] += dp[0][0][i][j]
        # フラグなし→↑
        dp[1][0][i+1][j] += dp[0][1][i][j]
        # フラグなし→→
        dp[0][1][i][j+1] += dp[0][1][i][j]
        # フラグあり↑↑
        dp[0][0][i+1][j] += dp[1][0][i][j]
        # フラグあり→→
        dp[0][1][i][j+1] += dp[1][1][i][j]

        for k in range(2):
            for l in range(2):
                dp[k][l][i][j] %= MOD

# (H-1, W-1)に辿り着いた全てのケースを合計する
ans = 0
for k in range(2):
    for l in range(2):
        ans += dp[k][l][H-1][W-1]
        ans %= MOD
print(ans)
