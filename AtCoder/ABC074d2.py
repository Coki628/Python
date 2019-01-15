# -*- coding: utf-8 -*-

"""
・ワーシャルフロイドの応用
・O(N^3)=300^3=2700万はTLEするからpypyで
・色々試行錯誤で結構時間かかったけど、500点の自力ACは嬉しい。
"""

from copy import deepcopy

N = int(input())
A = []
for i in range(N):
    A.append(list(map(int, input().split())))

# ワーシャルフロイドで全頂点の最短距離
B = deepcopy(A)
for k in range(N):
    for i in range(N):
        for j in range(N):
            # 最短 > 経由 があればNG
            if A[i][j] > A[i][k] + A[k][j]:
                print(-1)
                exit()
            # 最短 = 経由 があるなら、経由を使えば全長を減らせる
            elif A[i][j] == A[i][k] + A[k][j] and j != k and i != k and i < j:
                # 最短経路(仮)から直行ルートをなくす
                B[i][j] = 0

# 削減された最短経路の全長
sm = 0
for i in range(N):
    for j in range(i+1, N):
        sm += B[i][j]
print(sm)
