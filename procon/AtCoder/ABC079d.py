# -*- coding: utf-8 -*-

"""
自力AC
"""

from scipy.sparse.csgraph import floyd_warshall

H, W = map(int, input().split())
C = [None] * 10
for i in range(10):
    C[i] = list(map(int, input().split()))
field = [None] * H
for i in range(H):
    field[i] = list(map(int, input().split()))

# ちょうどよく隣接行列なので、そのままscipyのワーシャルフロイド使う
dist = floyd_warshall(C)

ans = 0
for i in range(H):
    for j in range(W):
        if field[i][j] != -1:
            # 該当マスの数字から1への距離を足していくだけ
            ans += dist[field[i][j]][1]
print(int(ans))
