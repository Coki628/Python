# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・速攻自力AC
・重み付きUnion-Find
・まあライブラリで殴るだけだったけど、問題見た時にぱっとすぐこの方針が出たのは良かった。
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

class WeightedUnionFind:
    """ 重み付きUnion-Find木 """

    def __init__(self, n):
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        # 根への距離を管理
        self.weight = [0] * (n+1)

    def find(self, x):
        """ 検索 """
        if self.par[x] == x:
            return x
        else:
            y = self.find(self.par[x])
            # 親への重みを追加しながら根まで走査
            self.weight[x] += self.weight[self.par[x]]
            self.par[x] = y
            return y

    def union(self, x, y, w):
        """ 併合 """
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

    def same(self, x, y):
        """ 同じ集合に属するか """
        return self.find(x) == self.find(y)

    def diff(self, x, y):
        """ xからyへのコスト """
        return self.weight[x] - self.weight[y]

N, M = MAP()
wuf = WeightedUnionFind(N)
for i in range(M):
    l, r, d = MAP()
    # 既に同じ集合にあるなら、距離が正しいか確認
    if wuf.same(l, r):
        if wuf.diff(l, r) != d:
            No()
            exit()
    else:
        wuf.union(l, r, d)
Yes()
