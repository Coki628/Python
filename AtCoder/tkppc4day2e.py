# -*- coding: utf-8 -*-

"""
・自力AC！
・最小全域木っぽくUnionFindで連結させていく。
・途中で集合の要素数の確認が必要なので、集合の変化に合わせて
　配列の中身を動かして管理する。
・現在どこまでOKかを変数で持たせておけば、それは前に進むだけなので、
　内側のwhile文で進める処理しても計算量は2乗にならない。
"""

import sys
from operator import itemgetter
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

class UnionFind:
    """ Union-Find木 """

    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        self.size = [1] * (n+1)
        self.tree = [True] * (n+1)
        self.grpcnt = n

    def find(self, x):
        """ 根の検索 """
        t = []
        while self.par[x] != x:
            t.append(x)
            x = self.par[x]
        for i in t:
            self.par[i] = x
        return self.par[x]

    def union(self, x, y):
        """ 併合 """
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
        """ 同じ集合に属するか判定 """
        return self.find(x) == self.find(y)

    def get_size(self, x=None):
        if x is not None:
            """ あるノードの属する集合のノード数 """
            return self.size[self.find(x)]
        else:
            """ 集合の数 """
            return self.grpcnt
    
    def is_tree(self, x):
        """ 木かどうかの判定 """
        return self.tree[self.find(x)]

N, M, Q = MAP()
edges = []
for _ in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    edges.append((a, b, c))

edges.sort(key=itemgetter(2))
# C[i] := 集合の要素数がiの集合に属する要素数
C = Counter()
C[1] = N
# need[i] := 全員が人数iを達成するために必要なコスト
need = [INF] * (N+1)
need[0] = need[1] = 0
uf = UnionFind(N)
# cur := 現在達成できている人数
cur = 1
for a, b, c in edges:
    if not uf.is_same(a, b):
        acnt = uf.get_size(a)
        bcnt = uf.get_size(b)
        uf.union(a, b)
        # 今回のマージによる集合の変化
        C[acnt] -= acnt
        C[bcnt] -= bcnt
        C[acnt+bcnt] += acnt + bcnt
        # 現在位置が0になったら、進めるところまで進む
        while C[cur] == 0:
            cur += 1
            need[cur] = c

ans = []
for _ in range(Q):
    q = INT()
    if q > N or need[q] == INF:
        ans.append('trumpet')
    else:
        ans.append(str(need[q]))
print('\n'.join(ans))
