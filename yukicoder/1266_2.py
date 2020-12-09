"""
・自力AC！こういうグラフ頑張るのは好き。
・グラフ、頂点拡張、UnionFind
・色についても含めて[都市][色]の組を1つの頂点として扱う。
・隣り合う色、同じ色で辺のある都市同士、は連結として、色が増えた時も都度周辺を確認して連結させる。
　回答クエリには連結成分のサイズでOK。
・なんか同じ頂点から辺がたくさん生えてると計算量がやばくなりそうな気がして、
　辺はsetで管理して連結になった時点でそこは削除、みたいにやったんだけど、別にやらなくてよかった。。
・setのところlistにした版。こっちのが全然速い。結局1つの辺は(色の区別、往復はあるものの)
　1回しか通らないもんね。。pypyAC2.49→1.58秒。(制約3秒)
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

N, M, Q = MAP()
L = 7
uf = UnionFind(N*L)
C = [[]] * N
for i in range(N):
    C[i] = list(map(int, input()))
    for j in range(L):
        if C[i][j] and C[i][(j+1)%L]:
            uf.union(i*L+j, i*L+(j+1)%L)
nodes = [[[] for j in range(7)] for i in range(N)]
for i in range(M):
    u, v = MAP()
    u -= 1; v -= 1
    for j in range(L):
        if C[u][j] and C[v][j]:
            uf.union(u*L+j, v*L+j)
        else:
            nodes[u][j].append(v)
            nodes[v][j].append(u)

for _ in range(Q):
    op, x, c = MAP()
    x -= 1; c -= 1
    if op == 1:
        C[x][c] = 1
        if C[x][(c-1)%L]:
            uf.union(x*L+c, x*L+(c-1)%L)
        if C[x][(c+1)%L]:
            uf.union(x*L+c, x*L+(c+1)%L)
        for v in nodes[x][c]:
            if C[v][c]:
                uf.union(x*L+c, v*L+c)
    else:
        res = uf.get_size(x*L)
        print(res)
