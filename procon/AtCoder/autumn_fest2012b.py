# -*- coding: utf-8 -*-

"""
・グリッド、UnionFind
・グリッドをグラフとみなしてUnionFindで連結を管理する。
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
    
    def get_group(self):
        """ グループ番号の取得 """
        res = set()
        for i in range(self.n+1):
            res.add(self.find(i))
        return res

    def get_size(self, x=None):
        if x is not None:
            """ あるノードの属する集合のノード数 """
            return self.size[self.find(x)]
        else:
            """ 集合の数 """
            return len(self.get_group()) - 1
    
    def is_tree(self, x):
        """ 木かどうかの判定 """
        return self.tree[self.find(x)]

def build_grid(H, W, intv, _type, space=True, padding=False):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

H, W = MAP()
grid = build_grid(H, W, 0, int, space=0)

# グリッドをグラフとみなしてマス同士を連結させる
uf = UnionFind(H*W)
for i in range(H):
    for j in range(1, W-1):
        if grid[i][j-1] == grid[i][j] == grid[i][j+1]:
            uf.union(i*W + j-1, i*W + j)
            uf.union(i*W + j, i*W + j+1)
for j in range(W):
    for i in range(1, H-1):
        if grid[i-1][j] == grid[i][j] == grid[i+1][j]:
            uf.union((i-1)*W + j, i*W + j)
            uf.union(i*W + j, (i+1)*W + j)

# 各集合の情報を取得して、カタマリになっているものを確認
groups = sorted(uf.get_group())[:-1]
N = len(groups)
is_bulk = set()
for group in groups:
    if uf.get_size(group) > 1:
        is_bulk.add(group)

# カタマリになった隣り合う集合同士の連結を確認する
for i in range(H):
    for j in range(W-1):
        a, b = uf.find(i*W + j), uf.find(i*W + j+1)
        if a in is_bulk and b in is_bulk and grid[i][j] == grid[i][j+1]:
            uf.union(i*W + j, i*W + j+1)
for j in range(W):
    for i in range(H-1):
        a, b = uf.find(i*W + j), uf.find((i+1)*W + j)
        if a in is_bulk and b in is_bulk and grid[i][j] == grid[i+1][j]:
            uf.union(i*W + j, (i+1)*W + j)

# 各集合の情報を再取得して、改めてカタマリの数を数える
groups = sorted(uf.get_group())[:-1]
cnt = 0
for group in groups:
    if uf.get_size(group) > 1:
        cnt += 1
print(cnt)
