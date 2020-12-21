"""
参考：https://yukicoder.me/problems/no/583/editorial
　　　https://www.hamayanhamayan.com/entry/2017/10/28/010508
・自力ならず。まあ、知らなきゃしゃあないか。。
・グラフ、一筆書きの判定、オイラー路
・一筆書きでパスが書けるかどうかは、辺を持つ全ての頂点が連結かどうかと、
　次数が奇数になる頂点が0か2であることの2つが条件になる。
・この一筆書きのパスのことを、それっぽい用語ではオイラー路と言うそう。
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

N, M = MAP()
nodes = [[] for i in range(N)]
uf = UnionFind(N)
for i in range(M):
    a, b = MAP()
    nodes[a].append(b)
    nodes[b].append(a)
    uf.union(a, b)

oddcnt = 0
se = set()
for i in range(N):
    if len(nodes[i]) % 2 == 1:
        oddcnt += 1
    if nodes[i]:
        se.add(uf.find(i))

if len(se) == 1 and oddcnt in [0, 2]:
    YES()
else:
    NO()
