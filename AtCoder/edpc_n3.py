# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-10
・区間DP、累積和
・普通のDPでの区間DP、練習のためにこれでもやってみた。
・ちょっと速くなってpypy0.5秒AC。
"""

import sys
from itertools import accumulate

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

N = INT()
A = LIST()

# 累積和
acc = [0] + list(accumulate(A))

# dp[l][r] := 区間[l,r)のスライムを合体させる際に必要な最小コスト
dp = list2d(N+1, N+1, INF)
for i in range(N):
    dp[i][i+1] = 0
for d in range(2, N+1):
    for l in range(N+1-d):
        r = l + d
        for m in range(l+1, r):
            dp[l][r] = min(dp[l][r], dp[l][m]+dp[m][r])
        # この先切った分の最小値に、今回のlとrを合体させる分を足す
        dp[l][r] += acc[r] - acc[l]
print(dp[0][N])
