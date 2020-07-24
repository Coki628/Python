"""
・青diff600点自力AC！嬉しい。
・グラフ、木、森、UnionFind
・観察すると、グラフを連結とするには辺が「集合の数-1」本必要で、各頂点1回しか使えない条件なので、
　頂点は「(集合の数-1)*2」個必要となる。これが足りなければNG。0なら0でOK。
・各頂点をどこから選べるかを考えることになるが、各集合から最低1個は絶対必要、残りはどこから
　取ってきてもいい、が正解。一見適当に選ぶと全部がうまく繋がらないような気になってしまうが、
　使う頂点は決まっていても、どれとどれを繋いで辺とするかは自由に決められるので、
　各集合から最低1個、の条件さえ守っていれば、よしなに繋げばうまく1つの木にできる。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

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
A = LIST()
uf = UnionFind(N)
for i in range(M):
    a, b = MAP()
    uf.union(a, b)

# 必要な頂点の本数K
L = uf.get_size()
K = (L-1) * 2
# 例外処理
if K > N:
    print('Impossible')
    exit()
elif K == 0:
    print(0)
    exit()

# 各集合毎にまとめる
adjli = [[] for i in range(N)]
for i in range(N):
    root = uf.find(i)
    adjli[root].append(A[i])

ans = 0
B = []
for i in range(N):
    if adjli[i]:
        # 各集合から1つ(最小コストのもの)は必ず使う
        adjli[i].sort(reverse=1)
        ans += adjli[i].pop()
        B += adjli[i]
        K -= 1
# 残りは無条件にコストが小さい方から取る
B.sort()
ans += sum(B[:K])
print(ans)
