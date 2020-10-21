"""
・なんとか自力AC。結構悩んだ。
・木、グラフ
・最初は1直線グラフに貪欲につないでくのとか考えたんだけどうまくいかなくて、
　図書いてどうするかぐりぐり考えてたら、1人が自分と違うグループのやつ全員と繋がれば、
　自分と同じグループのやつ以外は全てOKになると気付く。同じように、
　違うグループのやつを1人用意して、そいつはさっきの最初のやつと同じグループのやつ全員と繋がれば、
　これで全員連結になるんじゃないかと思う。こうなると、全員同じグループの場合以外は全部可能になる。
・この考察は合ってて、無事AC。でもこの問題、制約がN^2を許すので、
　全ペアについてつなげるかチェックしながらUnionFindで連結判定して繋がってないとこ全部繋ぐ、
　でもよかったみたい。
・UnionFindの方針もやってみた。でも多分5000^2きつくてTLE。。
　N^2許すならなんでこんな制約攻めたんだろう。
"""

import sys

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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

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

for _ in range(INT()):
    N = INT()
    A = LIST()

    uf = UnionFind(N)
    ans = []
    for i in range(N):
        for j in range(i+1, N):
            if A[i] != A[j]:
                if not uf.is_same(i, j):
                    uf.union(i, j)
                    ans.append((i+1, j+1))

    if uf.get_size() != 1:
        NO()
    else:
        YES()
        for a in ans:
            print(*a)
