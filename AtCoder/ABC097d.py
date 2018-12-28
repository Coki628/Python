# -*- coding: utf-8 -*-

"""
参考：http://at274.hatenablog.com/entry/2018/02/02/173000
・Union-Find木
・これである頂点間に経路が繋がっているかどうか素早く分かる。
"""

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
    def same_check(self, x, y):
        return self.find(x) == self.find(y)


N, M = map(int, input().split())
pN = list(map(int, input().split()))
uf = UnionFind(N)
for i in range(M):
    x, y = map(int, input().split())
    uf.union(x, y)

cnt = 0
for i in range(N):
    if uf.same_check(pN[i], i+1):
        cnt += 1
print(cnt)
