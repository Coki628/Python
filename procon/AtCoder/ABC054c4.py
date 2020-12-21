# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・別解、bitDP、TSP数え上げ
・一応練習のため。制約がN=15くらいあればこっち。
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

N, M = MAP()
G = list2d(N, N, 0)
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    G[a][b] = G[b][a] = 1

# dp[S][i] := 訪問済の頂点集合をSとして、最後に頂点iにいる場合の通り数
dp = list2d(1<<N, N, 0)
# 最初は頂点1にいる状態から
dp[1][0] = 1
for S in range(1, 1<<N):
    for i in range(N):
        # iが遷移元に含まれていない
        if not S & 1<<i:
            continue
        for j in range(N):
            # jが遷移元に含まれている(ので遷移先にはできない)
            if S & 1<<j:
                continue
            # 辺の張られた頂点間であれば遷移させる
            if G[i][j]:
                dp[S|1<<j][j] += dp[S][i]
# 全頂点訪問できた通り数を合計する
ans = sum(dp[(1<<N)-1])
print(ans)
