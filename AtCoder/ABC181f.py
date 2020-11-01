"""
参考：https://twitter.com/50j1_/status/1322896938300157953
・自力ならず。考察立たず。まあ残り時間も短すぎたけど。。
・幾何と見せかけて本質はグラフに帰着、UnionFind、小数の二分探索
・通れない方に注目すると、通れない2点間に壁(辺)を張っていって、上下が繋がったらNG、と解釈できる。
　あとはどこまで辺を張っていいかで二分探索する。
"""

import sys
from math import hypot

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

def bisearch_max(mn, mx, func, times):
    """ 条件を満たす最大値を見つける二分探索(小数用) """

    ok = mn
    ng = mx
    for _ in range(times):
        mid = (ok+ng) / 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

N = INT()
XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))

def check(m):
    uf = UnionFind(N+2)

    # 上端〜各点
    for i in range(N):
        if hypot(0, XY[i][1]-100) < m:
            uf.union(N, i)
    # 各点〜下端
    for i in range(N):
        if hypot(0, XY[i][1]+100) < m:
            uf.union(N+1, i)

    for i in range(N):
        for j in range(i+1, N):
            # 各2点間
            if hypot(XY[i][0]-XY[j][0], XY[i][1]-XY[j][1]) < m:
                uf.union(i, j)

    # 上限が繋がってなければOK
    return not uf.is_same(N, N+1)

res = bisearch_max(0, 200, check, 100) / 2
print(res)
