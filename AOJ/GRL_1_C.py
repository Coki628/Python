# -*- coding: utf-8 -*-

"""
・全点対間最短距離
・ワーシャルフロイド
"""

import sys
from copy import deepcopy

sys.setrecursionlimit(10 ** 9)
def input(): return sys.stdin.readline().strip()
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
INF=float('inf')

# ワーシャルフロイド(頂点数, 隣接行列(0-indexed))
def warshall_floyd(N: int, graph: list) -> list:
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

N,M=MAP()
G=[[INF]*N for i in range(N)]
for i in range(M):
    s,t,d=MAP()
    G[s][t]=d

ans=warshall_floyd(N, G)

if not len(ans):
    print('NEGATIVE CYCLE')
    exit()
for i in range(N):
    for j in range(N):
        if ans[i][j]==INF:
            ans[i][j]='INF'
for i in range(N):
    print(*ans[i])
