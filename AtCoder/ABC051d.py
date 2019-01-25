# -*- coding: utf-8 -*-

"""
・速攻自力AC
・ワーシャルフロイド
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = map(int, input().split())

G = [[float('inf')] * N for i in range(N)]
edges = [None] * M
for i in range(M):
    a, b, c = map(int, input().split())
    G[a-1][b-1] = c
    G[b-1][a-1] = c
    edges[i] = (a-1, b-1, c)

# ワーシャルフロイドで全頂点の最短距離
for k in range(N):
    for i in range(N):
        for j in range(N):
            if i == j:
                G[i][j] = 0
            else:
                G[i][j] = min(G[i][j], G[i][k] + G[k][j])

ans = 0
for a, b, c in edges:
    # 最短距離ではない辺の数を数える
    if G[a][b] < c:
        ans += 1
print(ans)
