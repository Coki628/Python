# -*- coding: utf-8 -*-

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

    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        self.size = [1] * (n+1)
        self.tree = [True] * (n+1)

    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            self.tree[x] = False
            return
        if not self.tree[x] or not self.tree[y]:
            self.tree[x] = self.tree[y] = False

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
            res = set()
            for i in range(self.n+1):
                res.add(self.find(i))
            return len(res) - 1
    
    def is_tree(self, x):
        return self.tree[self.find(x)]

N = INT()
segs = [set() for i in range(N*2+1)]
for i in range(N):
    l, r = MAP()
    segs[l].add(i)
    segs[r].add(i)

edges = set()
S1 = set()
for i, seg1 in enumerate(segs):
    for id1 in seg1:
        if id1 not in S1:
            S1.add(id1)
        else:
            j = i - 1
            S2 = set()
            while True:
                seg2 = segs[j]
                if id1 in seg2:
                    break
                for id2 in seg2:
                    if id2 not in S2:
                        S2.add(id2)
                    else:
                        S2.remove(id2)
                j -= 1
            for id2 in S2:
                edges.add((min(id1, id2), max(id1, id2)))
                if len(edges) >= N:
                    NO()
                    exit()
                
uf = UnionFind(N)
for a, b in edges:
    uf.union(a, b)

if uf.get_size() != 1 or not uf.is_tree(0):
    NO()
else:
    YES()
