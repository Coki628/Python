# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
・ワーシャルフロイド、bitDP(TSP)
・方針としてはjag2013autumn_cと同じ感じ。
・でも頂点8だから順列でN!しても良かった。。
・pythonTLE,pypyAC0.4秒。
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

def warshall_floyd(N: int, graph: list) -> list:
    """ ワーシャルフロイド(頂点数, 隣接行列(0-indexed)) """
    from copy import deepcopy

    res = deepcopy(graph)
    for i in range(N):
        # 始点 = 終点、は予め距離0にしておく
        res[i][i] = 0
    # 全頂点の最短距離
    for k in range(N):
        for i in range(N):
            for j in range(N):
                res[i][j] = min(res[i][j], res[i][k] + res[k][j])
    # 負の閉路(いくらでもコストを減らせてしまう場所)がないかチェックする
    for i in range(N):
        if res[i][i] < 0:
            return []
    return res

N, M, R = MAP()
A = [a-1 for a in LIST()]
G = list2d(N, N, INF)
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    G[a][b] = c
    G[b][a] = c

# 全体グラフの最短距離(ここから必要な頂点間だけ使う)
wf = warshall_floyd(N, G)

ans = INF
for r in range(R):
    # TSP(巡回セールスマン)
    dp = list2d(1<<R, R, INF)
    dp[1<<r][r] = 0
    for bit in range(1, (1<<R)-1):
        for i in range(R):
            if not (bit >> i & 1):
                continue
            for j in range(R):
                if bit >> j & 1:
                    continue
                a, b = A[i], A[j]
                dp[bit|1<<j][j] = min(dp[bit|1<<j][j], dp[bit][i] + wf[a][b])
    ans = min(ans, min(dp[-1]))
print(ans)
