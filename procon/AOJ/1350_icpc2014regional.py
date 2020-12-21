# -*- coding: utf-8 -*-

"""
参考：http://tsutaj.hatenablog.com/entry/2016/12/31/160154
・蟻本演習2-5-3
・最小全域木応用
・問題は要約すると、全ての最小全域木に含まれるような辺を列挙するってこと。
・計算量は約N*M^2/2で6200万くらい。TLE。。
・C++に写経してAC0.1秒。
"""

import sys
from operator import itemgetter

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
INF = float('inf')
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

    def same(self, x, y):
        """ 同じ集合に属するか判定 """
        return self.find(x) == self.find(y)

    def get_size(self, x):
        """ あるノードの属する集合のノード数 """
        return self.size[self.find(x)]
    
    def is_tree(self, x):
        """ 木かどうかの判定 """
        return self.tree[self.find(x)]

    def len(self):
        """ 集合の数 """
        res = set()
        for i in range(self.n+1):
            res.add(self.find(i))
        # グループ0の分を引いて返却
        return len(res) - 1

N, M = MAP()
edges = []
for i in range(M):
    a, b, c = MAP()
    edges.append((a, b, c))

# とりあえず最小全域木の1つを求める
edges.sort(key=itemgetter(2))
uf = UnionFind(N)
cost = 0
ids = []
for i, (a, b, c) in enumerate(edges):
    if not uf.same(a, b):
        uf.union(a, b)
        cost += c
        # 使った辺を覚えておく(これが必須かもしれない辺のリストになる)
        ids.append(i)
mncost = cost

needs = []
for i in ids:
    uf = UnionFind(N)
    cost = 0
    for j, (a, b, c) in enumerate(edges):
        # 辺iは使わない
        if i == j:
            continue
        if not uf.same(a, b):
            uf.union(a, b)
            cost += c
    # 辺iなしで構築した時に、木にならなかったor最小コストにならなかったら、iは必須
    if uf.len() != 1 or cost != mncost:
        needs.append(i)

ans = 0
for i in needs:
    ans += edges[i][2]
print(len(needs), ans)
