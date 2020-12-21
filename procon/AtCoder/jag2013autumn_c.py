# -*- coding: utf-8 -*-

"""
・蟻本演習3-4-1、自力AC！
・bitDP、ワーシャルフロイド
・グラフを必要な頂点(ラーメン屋のある駅と出発点)だけに変形してbitDPする。
・pythonTLE,pypyAC。
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
sys.setrecursionlimit(10 ** 7)
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

def bit_count(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

while True:
    N, M, L, S, T = MAP()
    S -= 1
    if N == 0:
        break

    G = list2d(N, N, INF)
    for i in range(M):
        a, b, c = MAP()
        G[a-1][b-1] = c
        G[b-1][a-1] = c
    # 全駅間の最短距離
    G = warshall_floyd(N, G)
    E = [0] * (L+1)
    # 必要な小さいグラフへのマッピング
    LtoM = {0: S}
    for i in range(L):
        l, e = MAP()
        E[i+1] = e
        LtoM[i+1] = l - 1
    # ラーメン屋のある駅と出発点だけのグラフにする
    L += 1
    G2 = list2d(L, L, INF)
    for i in range(L):
        for j in range(L):
            G2[i][j] = G[LtoM[i]][LtoM[j]]

    # TSP(巡回セールスマン)
    dp = list2d(1<<L, L, INF)
    dp[1][0] = 0
    for bit in range(1, (1<<L)-1):
        for i in range(L):
            if not (bit >> i & 1):
                continue
            for j in range(L):
                if bit >> j & 1:
                    continue
                # 駅間の距離と食べる時間を足して遷移
                dp[bit|1<<j][j] = min(dp[bit|1<<j][j], dp[bit][i]+G2[i][j]+E[j])

    ans = 0
    for bit in range(1, (1<<L)):
        # 回った駅の数 - 出発点
        cnt = bit_count(bit) - 1
        for i in range(L):
            if not (bit >> i & 1):
                continue
            # 駅Sに戻ってくる分
            dist = dp[bit][i] + G2[i][0]
            # T以内なら、答えに使える
            if dist <= T:
                ans = max(ans, cnt)
    print(ans)
