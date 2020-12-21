# -*- coding: utf-8 -*-

"""
・ワーシャルフロイド
・順列
・これは結構さくっと方針も決まって自力AC
・O(N^3)=200^3=800万でもTLEしたからpypyで
"""

from itertools import permutations

N, M, R = map(int, input().split())
r = list(map(int, input().split()))
G = [[float('inf')] * (N) for _ in range(N)]
for i in range(M):
    a, b, c = map(int, input().split())
    # 経路情報は隣接行列に保持する(無向グラフなので両方向)
    G[a-1][b-1] = c
    G[b-1][a-1] = c

# ワーシャルフロイドで全頂点の最短距離
for k in range(N):
    for i in range(N):
        for j in range(N):
            # 始点 = 終点、は例外的に距離0にしておく
            if i == j:
                G[i][j] = 0
            else:
                G[i][j] = min(G[i][j], G[i][k] + G[k][j])

ans = float('inf')
# 町の順番は順列全部試す(最大8なら大丈夫)
for p in permutations(r):
    sm = 0
    # 最短経路を足し合わせていく
    for i in range(1, len(p)):
        sm += G[p[i-1]-1][p[i]-1]
    ans = min(sm, ans)
print(ans)
