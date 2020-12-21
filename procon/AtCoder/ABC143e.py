# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc143/editorial.pdf
・ワーシャルフロイドを2回やる。
・2回目は燃料1回で行ける各2点に辺を張る。
・聞けば納得。がしかしこれを導き出す考察力が。。。
・計算量N^3=2700万、pythonTLEでpypyAC1.0秒。
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
    return res

N, M, L = MAP()
G = list2d(N, N, INF)
for i in range(M):
    a, b, c = MAP()
    a -= 1
    b -= 1
    G[a][b] = c
    G[b][a] = c

res = warshall_floyd(N, G)

G2 = list2d(N, N, INF)
for i in range(N):
    for j in range(i, N):
        # 最短距離がL以下の2点なら燃料1回で行ける
        if res[i][j] <= L:
            G2[i][j] = 1
            G2[j][i] = 1

# 燃料使用回数をコストとした最短経路
res2 = warshall_floyd(N, G2)

for i in range(INT()):
    a, b = MAP()
    a -= 1
    b -= 1
    ans = res2[a][b]
    if ans != INF:
        print(ans-1)
    else:
        print(-1)
