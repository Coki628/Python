# -*- coding: utf-8 -*-

"""
・自力AC！
・整数のグラフ、ワーシャルフロイド、ダイクストラ応用
・整数の各値を頂点としてグラフを構築するやつ。(ABC077Dを思い出した)
・計算量は、前にいた数字として0~9を添字に追加しているので、(最初これなかったらWAした)
　10万*10=100万頂点のダイクストラって感じになる。
・python,pypyはTLEしたのでC++でAC0.5秒。
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

def warshall_floyd(graph: list) -> list:
    """ ワーシャルフロイド(頂点数, 隣接行列(0-indexed)) """
    from copy import deepcopy

    N = len(graph)
    res = deepcopy(graph)
    for i in range(N):
        res[i][i] = 0
    for k in range(N):
        for i in range(N):
            for j in range(N):
                res[i][j] = min(res[i][j], res[i][k] + res[k][j])
    for i in range(N):
        if res[i][i] < 0:
            return []
    return res

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    # res[i][j] := 今いるキーがiで、入力した数値がjの時の最小コスト
    res = list2d(10, N, INF)
    que = [(0, src, 0)]
    res[0][src] = 1
    while que:
        dist, cur, a = heappop(que)
        if res[a][cur] < dist:
            continue
        # 0~9のキーを押す
        for nxta in range(10):
            nxt = (cur*10 + nxta) % M
            cost = wf[a][nxta] + 1
            if dist + cost < res[nxta][nxt]:
                res[nxta][nxt] = dist + cost
                heappush(que, (dist+cost, nxt, nxta))
    return res

M, R = MAP()

# 各キー間の移動距離を前計算
G = list2d(10, 10, INF)
G[0][1] = G[1][0] = 1
G[1][2] = G[2][1] = G[1][4] = G[4][1] = 1
G[2][3] = G[3][2] = G[2][5] = G[5][2] = 1
G[3][6] = G[6][3] = 1
G[4][7] = G[7][4] = G[4][5] = G[5][4] = 1
G[5][8] = G[8][5] = G[5][6] = G[6][5] = 1
G[6][9] = G[9][6] = 1
G[7][8] = G[8][7] = 1
G[8][9] = G[9][8] = 1
wf = warshall_floyd(G)

nodes = [[] for i in range(M)]
res = dijkstra(nodes, 0)
ans = INF
for i in range(10):
    ans = min(ans, res[i][R])
print(ans)
