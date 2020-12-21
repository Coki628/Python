"""
・なんとか自力AC。。
・かなり強引に適当に通した感があったからシステス無理だと思ったけど何とかなった。。
・グリッドをグラフに帰着させてUFする、まあ結構よくあるやつ。
　なんだけど条件に合ったいい感じの繋ぎ方が見えてこなくてつらかった。。
・後でちょっとまともな解法ちゃんと考えてまとめよう。。
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
    used = [0] * N
    for i in range(M):
        h, w = MAP()
        h -= 1; w -= 1
        if h == w:
            used[h] = 1
            continue
        rowcnt[h].append(i)
        colcnt[w].append(i)

    uf = UnionFind(M)
    for i in range(N):
        for j in range(len(rowcnt[i])-1):
            uf.union(rowcnt[i][j], rowcnt[i][j+1])
        for j in range(len(colcnt[i])-1):
            uf.union(colcnt[i][j], colcnt[i][j+1])
        if rowcnt[i] and colcnt[i]:
            uf.union(colcnt[i][0], rowcnt[i][0])

    cellcnt = [0] * M
    for i in range(N):
        if used[i]:
            continue
        if rowcnt[i]:
            cellcnt[uf.find(rowcnt[i][0])] += 1
        elif colcnt[i]:
            cellcnt[uf.find(colcnt[i][0])] += 1

    ans = 0
    for root in uf.get_roots():
        if uf.get_size(root) >= 2:
            ans += uf.get_size(root) + max((uf.get_size(root) - cellcnt[root]), 0) + (not uf.is_tree(root))
        else:
            ans += 1
    ans -= sum(used)
    print(ans)
