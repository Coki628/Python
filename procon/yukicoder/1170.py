"""
・自力AC！
・二分探索、UnionFind
・相互に到達できる駅数を数えるので、UFで連結判定ができそう。
　でもそれぞれの駅から範囲チェックをして有効な範囲全てを陽に連結させようとすると辺が多すぎる。
　どうにか無駄なところを少なくできないか。
・とりあえず、今見ている駅より右側だけ見ることにする。双方向に同じ感じにやるので、
　右へのチェックさえしっかりやっていれば、逆から左へのチェックはやる必要がないはず。
　これでにぶたんの方向が右だけになったのでbisectもやりやすくなった。
・次に、できるだけ遠くから繋げていくと、連結済まで来たら打ち切る、でうまくいきそう。
　そこより手前は自分より左の頂点をチェックした時すでに繋いであるはずだから。
　有効な範囲は1つの区間になっているので、隙間に中途半端な漏れが発生することはないはず、多分。
・これで実装して投げたら無事AC。
・似たようなUFの連結打ち切りをACL1_aでやってたのがヒントになったかな。
"""

import sys
from bisect import bisect_left, bisect_right

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

N, a, b = MAP()
A = LIST()

uf = UnionFind(N)
for i in range(N):
    l = bisect_left(A, A[i]+a, lo=i)
    r = bisect_right(A, A[i]+b, lo=i)
    # 有効な範囲を後ろから見ていって、連結済まで来たらそこで打ち切る
    for j in range(r-1, l-1, -1):
        if not uf.is_same(i, j):
            uf.union(i, j)
        else:
            break

for i in range(N):
    ans = uf.get_size(i)
    print(ans)
