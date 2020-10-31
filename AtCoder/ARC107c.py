"""
参考：みんなのツイート
・自力ならず。また難しく考えてしまった。。。
・グリッド行列スワップ、縦横独立、UnionFind、数え上げ
・縦横独立は疑ったんだけど、なんかダメな気がしちゃったんだよなー。
　でも確かにいくら行スワップしても、列単位で見た時に左右の関係変わらないね。。
・それが分かれば、連結成分毎にまとめて、同じグループ同士は順番を自由にできるから階乗、でOK。
　これ解けるべきだったよなぁ。。最近こんなことばっか言ってる気がしてへこむわ。
"""

import sys
from math import factorial

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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
MOD = 998244353
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

        return set([self.find(i) for i in range(N)])

N, K = MAP()
grid = [[]] * N
for i in range(N):
    grid[i] = LIST()

uf1 = UnionFind(N)
for i in range(N):
    for j in range(i+1, N):
        ok = True
        for k in range(N):
            if grid[i][k] + grid[j][k] > K:
                ok = False
        if ok:
            uf1.union(i, j)

uf2 = UnionFind(N)
for i in range(N):
    for j in range(i+1, N):
        ok = True
        for k in range(N):
            if grid[k][i] + grid[k][j] > K:
                ok = False
        if ok:
            uf2.union(i, j)

ans = 1
for root in uf1.get_roots():
    ans *= factorial(uf1.get_size(root))
    ans %= MOD
for root in uf2.get_roots():
    ans *= factorial(uf2.get_size(root))
    ans %= MOD
print(ans)
