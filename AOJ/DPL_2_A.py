# -*- coding: utf-8 -*-

"""
・巡回セールスマン問題
・bitDP
・これはTLE
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

N, M = MAP()
G = list2d(N, N, INF)
for i in range(M):
    s, t, d = MAP()
    G[s][t] = d

# dp[bit][i][j] := iからスタートして今jにいる時の、集合bitの最短距離
dp = list3d(1<<N, N, N, INF)
for i in range(N):
    # 初期化：頂点1つだけの各集合を開始位置として0にする
    dp[1<<i][i][i] = 0

for start in range(N):
    for bit in range(1, (1<<N)-1):
        for i in range(N):
            # 頂点iが集合bitに含まれていない
            if not bit>>i&1: continue
            for j in range(N):
                # 頂点jは既に集合bitに含まれている
                if bit>>j&1: continue
                # 頂点iからjへの移動を遷移させる
                dp[bit+(1<<j)][start][j] = min(dp[bit+(1<<j)][start][j], dp[bit][start][i] + G[i][j])

mn=INF
for start in range(N):
    for end in range(N):
        # 最後の頂点から戻ってくる分を足す
        mn = min(mn, dp[(1<<N)-1][start][end] + G[end][start])
if mn==INF:
    print(-1)
else:
    print(mn)
