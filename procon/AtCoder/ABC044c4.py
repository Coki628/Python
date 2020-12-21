# -*- coding: utf-8 -*-

"""
参考：http://arc060.contest.atcoder.jp/data/arc/060/editorial.pdf
・蟻本演習3-1-4
・これは愚直3次元DP解。計算量はN^3*K=50^4=625万。pythonTLE、pypyAC0.3秒。
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
sys.setrecursionlimit(10 ** 7)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N, K = MAP()
A = LIST()

MAX = N * K
# dp[i][j][k] := i枚目までからj枚取って、合計をkにする通り数
dp = list3d(N+1, N+1, MAX+1, 0)
dp[0][0][0] = 1
for i in range(N):
    a = A[i]
    for j in range(i+1):
        for k in range(MAX+1):
            dp[i+1][j][k] += dp[i][j][k]
            if j+1 <= N and k+a <= MAX:
                dp[i+1][j+1][k+a] += dp[i][j][k]
ans = 0
for j in range(1, N+1):
    # 1枚以上取った時の、平均がKになる所 → j枚取って合計がj*Kの所 を合計する
    ans += dp[N][j][j*K]
print(ans)
