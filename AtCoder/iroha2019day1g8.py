# -*- coding: utf-8 -*-

"""
・この後のnumpy実装練習のやりやすさのために、配るDP版を準備。
・この時点で計算量N^3=365^3=4900万くらい。
　pypyでギリギリも頷ける。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M, K = MAP()
A = [0] + LIST() + [0]*K

# dp[i][j] := j日目にi回使った状態
dp=list2d(M+1, N+K+1, -INF)
dp[0][0] = 0
for i in range(M):
    for j in range(N):
        for k in range(j+1, j+K+1):
            # 1~K日後の範囲内で全ての日に遷移してみる
            dp[i+1][k] = max(dp[i+1][k], dp[i][j]+A[k])

# M回終わって、あとK日で終わりまで行ければOK
ans = -INF
for i in range(N-K+1, N+1):
    ans = max(ans, dp[M][i])
if ans != -INF:
    print(ans)
else:
    print(-1)
