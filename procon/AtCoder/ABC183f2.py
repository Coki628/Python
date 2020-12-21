"""
・自力AC！
・マージテク、UnionFind
・上級界隈でよく見かけるマージテクだけど、本質で問われるのは珍しい印象。
・集合をマージする時、両方の集合が管理するCounterもマージする訳だけど、
　この時種類数の多い方をベースにして、多い方に少ない方をくっつけるようにすると、
　計算量が最終的にlogで済むようになる。
・手持ちが経路圧縮するタイプのUFなので、都度Counterの大小を確認しながらやったけど、
　TLの情報から、どうやらこっちのUFでも普通に親になった方にマージしてくを繰り返せば
　よしなにマージテクと同じことになるらしい。
　ってことでやってみた。問題なくpypyAC。1.17→0.99でちょっと速くなったしね。
"""

import sys
from collections import Counter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class UnionFind:
    """ Union-Find木 """

    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n)]
        self.rank = [0] * n
        self.size = [1] * n
        self.tree = [True] * n
        self.grpcnt = n

    def find(self, x):
        """ 根の検索(グループ番号の取得) """
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

    def get_roots(self):
        """ 全ての根を取得 """

        return set([self.find(i) for i in range(self.n)])

N, Q = MAP()
A = [a-1 for a in LIST()]

def merge(C1, C2):
    for k, v in C2.items():
        C1[k] += v
    return C1

C = [Counter([A[i]]) for i in range(N)]
uf = UnionFind(N)
for _ in range(Q):
    op, a, b = MAP()
    a -= 1; b -= 1
    if op == 1:
        if not uf.is_same(a, b):
            ar, br = uf.find(a), uf.find(b)
            uf.union(a, b)
            # 今回親になった方に向かってCounterもマージする
            if uf.find(a) == ar:
                C[ar] = merge(C[ar], C[br])
            else:
                C[br] = merge(C[br], C[ar])
    else:
        res = C[uf.find(a)][b]
        print(res)
