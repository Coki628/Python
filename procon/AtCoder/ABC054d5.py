# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・さくっと自力AC！
・ナップザック系で1つ項目が多い3次元DP
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

N, X, Y = MAP()
ABC = []
for i in range(N):
    a, b, c = MAP()
    ABC.append((a, b, c))

MAX = 407
# dp[i][j][k] := i個目まで見て、物質Aをj、物質Bをk含んだ状態での最小コスト
dp = list3d(N+1, MAX, MAX, INF)
dp[0][0][0] = 0
for i in range(N):
    a, b, c = ABC[i]
    for j in range(MAX):
        for k in range(MAX):
            dp[i+1][j][k] = min(dp[i+1][j][k], dp[i][j][k])
            if j+a < MAX and k+b < MAX:
                dp[i+1][j+a][k+b] = min(dp[i+1][j+a][k+b], dp[i][j][k] + c)

j = X
k = Y
ans = INF
while j < MAX and k < MAX:
    # N個見終わった状態で混合比がX:Yになるところから最小を取る
    ans = min(ans, dp[N][j][k])
    j += X
    k += Y
if ans == INF:
    print(-1)
else:
    print(ans)
