# -*- coding: utf-8 -*-

"""
参考：http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3687352#1
・巡回セールスマン問題
・bitDP
・出発点を頂点0だけにしたらAC
・問題文だと、どの出発点を選んでもいいってなってるのに
　なんでこれでACできるんだろー、って思ったけど、
　最終的に最短の閉路を作ることになるから、
　出発点がどこだろうと最短距離は変わらないんだね多分。
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

# dp[bit][i] := 今iにいる時の、集合bitの最短距離
dp = list2d(1<<N, N, INF)
# 初期化：頂点0を開始位置として距離0にする
dp[1][0] = 0

for bit in range(1, (1<<N)-1):
    for i in range(N):
        # 頂点iが集合bitに含まれていない
        if not bit>>i&1: continue
        for j in range(N):
            # 頂点jは既に集合bitに含まれている
            if bit>>j&1: continue
            # 頂点iからjへの移動を遷移させる
            dp[bit+(1<<j)][j] = min(dp[bit+(1<<j)][j], dp[bit][i] + G[i][j])

mn = INF
for end in range(N):
    # 最後の頂点から戻ってくる分を足す
    mn = min(mn, dp[(1<<N)-1][end] + G[end][0])
if mn==INF:
    print(-1)
else:
    print(mn)
