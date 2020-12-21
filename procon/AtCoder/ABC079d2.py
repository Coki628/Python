# -*- coding: utf-8 -*-

"""
scipyライブラリなし版
"""

H, W = map(int, input().split())
C = [None] * 10
for i in range(10):
    C[i] = list(map(int, input().split()))
field = [None] * H
for i in range(H):
    field[i] = list(map(int, input().split()))

# ワーシャルフロイドで全頂点の最短距離
for k in range(10):
    for i in range(10):
        for j in range(10):
            C[i][j] = min(C[i][j], C[i][k] + C[k][j])

ans = 0
for i in range(H):
    for j in range(W):
        if field[i][j] != -1:
            # 該当マスの数字から1への距離を足していくだけ
            ans += C[field[i][j]][1]
print(ans)
