# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・自力AC、結構悩んだけど。。
・最小全域木
・縦横独立に考える。
・ソートすれば、隣り合う頂点間の辺だけ考えればいい。
　離れた頂点を繋ぐ辺は、複数の隣り合う頂点間の辺に置き換えられるので得はしない。
・終わってみれば、このくらいすぐ出ないかなぁって思うんだけどね。。
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

class UnionFind:
    """ Union-Find木 """

    def __init__(self, n):
        self.n = n
        # 親要素のノード番号を格納。par[x] == xの時そのノードは根
        # 1-indexedのままでOK、その場合は[0]は未使用
        self.par = [i for i in range(n+1)]
        # 木の高さを格納する（初期状態では0）
        self.rank = [0] * (n+1)
        # あるノードを根とする集合に属するノード数
        self.size = [1] * (n+1)
        # あるノードを根とする集合が木かどうか
        self.tree = [True] * (n+1)

    def find(self, x):
        """ 根の検索(グループ番号と言えなくもない) """
        # 根ならその番号を返す
        if self.par[x] == x:
            return x
        else:
            # 走査していく過程で親を書き換える
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    def union(self, x, y):
        """ 併合 """
        # 根を探す
        x = self.find(x)
        y = self.find(y)

        # 木かどうかの判定用
        if x == y:
            self.tree[x] = False
            return
        if not self.tree[x] or not self.tree[y]:
            self.tree[x] = self.tree[y] = False

        # 木の高さを比較し、低いほうから高いほうに辺を張る
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.size[y] += self.size[x]
        else:
            self.par[y] = x
            self.size[x] += self.size[y]
            # 木の高さが同じなら片方を1増やす
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def is_same(self, x, y):
        """ 同じ集合に属するか判定 """
        return self.find(x) == self.find(y)

    def get_size(self, x=None):
        if x is not None:
            """ あるノードの属する集合のノード数 """
            return self.size[self.find(x)]
        else:
            """ 集合の数 """
            res = set()
            for i in range(self.n+1):
                res.add(self.find(i))
            # グループ0の分を引いて返却
            return len(res) - 1
    
    def is_tree(self, x):
        """ 木かどうかの判定 """
        return self.tree[self.find(x)]

N = INT()
X = []
Y = []
for i in range(N):
    x, y = MAP()
    X.append((x, i))
    Y.append((y, i))

# xとy個別にソートして、距離の隣り合う頂点間だけ辺を貼る
X.sort()
Y.sort()
edges = []
for i in range(N-1):
    x1, u = X[i]
    x2, v = X[i+1]
    edges.append((x2-x1, u, v))
    y1, u = Y[i]
    y2, v = Y[i+1]
    edges.append((y2-y1, u, v))

# 最小全域木の構築
edges.sort()
uf = UnionFind(N)
ans = 0
for c, a, b in edges:
    if not uf.is_same(a, b):
        uf.union(a, b)
        ans += c
print(ans)
