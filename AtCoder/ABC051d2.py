# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
・ワーシャルフロイド
・方針はすぐ立ったけど、本当にそれで正しいか確証は持てなかった。証明AC。。
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

N, M = MAP()
G = list2d(N, N, INF)
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    G[a][b] = c
    G[b][a] = c

wf = warshall_floyd(N, G)

cnt = 0
for i in range(N):
    for j in range(i+1, N):
        # 辺があって、最短経路より長いなら使わない
        if G[i][j] != INF and G[i][j] > wf[i][j]:
            cnt += 1
print(cnt)
