# -*- coding: utf-8 -*-

"""
・最小全域木
・今回違ったけど、一応取っておこう。
"""

from copy import deepcopy
from operator import itemgetter

N = int(input())
A = []
for i in range(N):
    A.append(list(map(int, input().split())))

# ワーシャルフロイドで全頂点の最短距離
B = deepcopy(A)
for k in range(N):
    for i in range(N):
        for j in range(N):
            # 始点 = 終点、は例外的に距離0にしておく
            if i == j:
                B[i][j] = 0
            else:
                B[i][j] = min(B[i][j], B[i][k] + B[k][j])

# 情報が一致しているか確認
for i in range(N):
    for j in range(i+1, N):
        # 不一致があれば、情報より近い経路があったことになるのでNG
        if A[i][j] != B[i][j]:
            print(-1)
            exit()

# Union-Find木
class UnionFind:
    def __init__(self, n):
        # 親要素のノード番号を格納。par[x] == xの時そのノードは根
        # 1-indexedのままでOK、その場合は[0]は未使用
        self.par = [i for i in range(n+1)]
        # 木の高さを格納する（初期状態では0）
        self.rank = [0] * (n+1)

    # 検索
    def find(self, x):
        # 根ならその番号を返す
        if self.par[x] == x:
            return x
        else:
            # 走査していく過程で親を書き換える
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    # 併合
    def union(self, x, y):
        # 根を探す
        x = self.find(x)
        y = self.find(y)
        # 木の高さを比較し、低いほうから高いほうに辺を張る
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
        else:
            self.par[y] = x
            # 木の高さが同じなら片方を1増やす
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    # 同じ集合に属するか判定
    def same(self, x, y):
        return self.find(x) == self.find(y)

uf = UnionFind(N)

# データの持ち方を隣接行列から辺集合にする
edges = []
for i in range(N):
    for j in range(i+1, N):
        if i != j:
            edges.append((i, j, A[i][j]))
# コストでソートしておく
edges.sort(key=itemgetter(2))

# 最小全域木の作成
cost = 0
for x, y, w in edges:
    # 同じ集合に属するかどうかで閉路を作るかの判定になる
    if not uf.same(x, y):
        # 閉路を作らない辺を追加していく
        uf.union(x, y)
        cost += w
print(cost)
