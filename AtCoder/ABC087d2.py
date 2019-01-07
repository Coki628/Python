# -*- coding: utf-8 -*-

"""
参考：http://hamko.hatenadiary.jp/entry/2018/01/30/210057
　　　http://at274.hatenablog.com/entry/2018/02/03/140504
・重み付きUnion-Find木
"""

import sys
def input(): return sys.stdin.readline().strip()

class WeightedUnionFind:
    def __init__(self, n):
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        # 根への距離を管理
        self.weight = [0] * (n+1)

    # 検索
    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            y = self.find(self.par[x])
            # 親への重みを追加しながら根まで走査
            self.weight[x] += self.weight[self.par[x]]
            self.par[x] = y
            return y

    # 併合
    def union(self, x, y, w):
        rx = self.find(x)
        ry = self.find(y)
        # xの木の高さ < yの木の高さ
        if self.rank[rx] < self.rank[ry]:
            self.par[rx] = ry
            self.weight[rx] = w - self.weight[x] + self.weight[y]
        # xの木の高さ ≧ yの木の高さ
        else:
            self.par[ry] = rx
            self.weight[ry] = - w - self.weight[y] + self.weight[x]
            # 木の高さが同じだった場合の処理
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1

    # 同じ集合に属するか
    def same(self, x, y):
        return self.find(x) == self.find(y)

    # xからyへのコスト
    def diff(self, x, y):
        return self.weight[x] - self.weight[y]

N, M = map(int, input().split())

wuf = WeightedUnionFind(N)
for i in range(M):
    x, y, w = map(int, input().split())
    # 同じ集合に属する場合
    if wuf.same(x, y):
        # 記録された距離と今回の情報に矛盾がないか確認
        if wuf.diff(x, y) != w:
            print('No')
            exit()
    wuf.union(x, y, w)
# 全部無矛盾で木が構築できればOK
print('Yes')
