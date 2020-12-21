"""
・あとから落ち着いて整理した版
・同じ行列を共有する駒は1つもないので、よく考察すると、作られる各連結成分は閉路1つあるかないかになる。
　で、閉路がある連結成分は、駒数+1の操作が必要で、そうでなければ駒数、元々ゴールにいる分は引く。
　まとめると、駒の数 + 閉路の数 - 元々ゴールにいる数 でOK。
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
    rowcnt = [[] for i in range(N)]
    colcnt = [[] for i in range(N)]
    used = 0
    for i in range(M):
        h, w = MAP()
        h -= 1; w -= 1
        if h == w:
            used += 1
            continue
        rowcnt[h].append(i)
        colcnt[w].append(i)

    # 同じゴールを目指す駒同士を連結させる
    uf = UnionFind(M)
    for i in range(N):
        for j in range(len(rowcnt[i])-1):
            uf.union(rowcnt[i][j], rowcnt[i][j+1])
        for j in range(len(colcnt[i])-1):
            uf.union(colcnt[i][j], colcnt[i][j+1])
        if rowcnt[i] and colcnt[i]:
            uf.union(colcnt[i][0], rowcnt[i][0])

    # 駒の数 + 閉路の数 - 元々ゴールにいる数
    ans = M + sum(not uf.is_tree(root) for root in uf.get_roots()) - used
    print(ans)
