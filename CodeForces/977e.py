# -*- coding: utf-8 -*-

"""
・なんとか自力AC
・グラフ、UnionFind
・最初、サイクルっていうの見て本当のサイクルを全部集めてしまって失敗。。
・この問題でのサイクルは、1本道で円みたいになるやつだけ。
・詰まるところ、連結成分内で頂点数と辺の数が同じだといい。
"""

import sys
from collections import Counter

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

class UnionFind:

    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n)]
        self.rank = [0] * n
        self.size = [1] * n
        self.tree = [True] * n
        self.grpcnt = n

    def find(self, x):
        t = []
        while self.par[x] != x:
            t.append(x)
            x = self.par[x]
        for i in t:
            self.par[i] = x
        return self.par[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            self.tree[x] = False
            return
        if not self.tree[x] or not self.tree[y]:
            self.tree[x] = self.tree[y] = False

        self.grpcnt -= 1
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.size[y] += self.size[x]
        else:
            self.par[y] = x
            self.size[x] += self.size[y]
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def is_same(self, x, y):
        return self.find(x) == self.find(y)

    def get_size(self, x=None):
        if x is not None:
            return self.size[self.find(x)]
        else:
            return self.grpcnt
    
    def is_tree(self, x):
        return self.tree[self.find(x)]

N, M = MAP()
uf = UnionFind(N)
edges = []
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    uf.union(a, b)
    edges.append((a, b))
    nodes[a].append(b)
    nodes[b].append(a)

roots = set()
for i in range(N):
    roots.add(uf.find(i))
edgecnt = [0] * N
for a, b in edges:
    edgecnt[uf.find(a)] += 1

nodesset = [set() for i in range(N)]
for i in range(N):
    nodesset[uf.find(i)].add(len(nodes[i]))

ans = 0
for root in roots:
    if not uf.is_tree(root):
        if edgecnt[root] == uf.get_size(root) and len(nodesset[root]) == 1 and list(nodesset[root])[0] == 2:
            ans += 1
print(ans)
