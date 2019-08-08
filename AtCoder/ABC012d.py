# -*- coding: utf-8 -*-

"""
・さくっとD自力AC
・ワーシャルフロイド一撃
・計算量N^3=300^3=2700万はpythonTLEでpypy0.8秒AC。
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

def deepcopy(li): return [x[:] for x in li]

def warshall_floyd(N: int, graph: list) -> list:
    """ ワーシャルフロイド(頂点数, 隣接行列(0-indexed)) """

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
G=list2d(N, N, INF)
for i in range(M):
    a,b,t=MAP()
    G[a-1][b-1]=t
    G[b-1][a-1]=t

WF=warshall_floyd(N, G)

ans=INF
for i in range(N):
    ans=min(ans, max(WF[i]))
print(ans)
