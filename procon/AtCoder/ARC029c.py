# -*- coding: utf-8 -*-

"""
・類題攻略(cf1245dより)
・最小全域木、頂点を追加して辺を工夫。
・細かい設定とか出力内容は違うけど、大元を表す擬似頂点を作って最小全域木、っていう流れは同じ。
・向こうは辺が200万くらいになるけど、こっちは20万なので制約的にpythonは通りやすい。
・それでもpython,pypy共にAC1.4秒。UnionFindが弱いんだろうな。。
・全域木が完成したらbreak、の処理はなくした方が速くなって上記結果。
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
INF = 10 ** 18
MOD = 10 ** 9 + 7

class UnionFind:

    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        self.size = [1] * (n+1)

    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

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

N, M = MAP()
edges = []
for i in range(N):
    c = INT()
    # 各頂点から交易所を表す擬似頂点Nへの辺
    edges.append((c, i, N))
for i in range(M):
    x, y, c = MAP()
    x -= 1
    y -= 1
    edges.append((c, x, y))

edges.sort(key=itemgetter(0))
uf = UnionFind(N+1)
total = 0
for c, a, b in edges:
    if not uf.is_same(a, b):
        uf.union(a, b)
        total += c
print(total)
