"""
参考：https://twitter.com/50j1_/status/1340706980923494400
　　　他、人々のツイート
・グリッド、グラフ
・行と列を頂点にして、行列の接点を辺とするやつ。たまに見るね。
　対象物が飛車っぽく同じ行列をどこまでも行ける設定の場合、この形のグラフを疑う必要は確かにあったね。。
　この構成のグラフでUFやっても、(頂点と辺の関係は逆だけど)自分でやったのと同じような連結成分になって、
　やっぱり、駒の数 + 閉路の数 - 元々ゴールにいる数 をやればOK。
・あと実装してないけど、行列分けないでN頂点にして、x[i] -> y[i] みたいな有向辺のグラフにして
　SCCとかで閉路チェックやっても同じことになるっぽい。この形は、対角線の各マスgrid[i][i]を頂点として、
　右に出て下に行く辺と、左に出て上に行く辺を繋ぐみたいなイメージにしたらああなるほどってなった。
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
INF = 10 ** 19
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

    def get_roots(self):

        return set([self.find(i) for i in range(self.n)])

for _ in range(INT()):
    N, M = MAP()
    used = 0
    uf = UnionFind(N*2)
    for i in range(M):
        h, w = MAP()
        h -= 1; w -= 1
        if h == w:
            used += 1
            continue
        uf.union(h, N+w)

    for i in range(N):
        uf.union(i, N+i)

    ans = M + sum(not uf.is_tree(root) for root in uf.get_roots()) - used
    print(ans)
