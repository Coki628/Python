# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/indeednow-finalb
・蟻本演習2-5-3
・最小全域木、クラスカル法
・これはスコアの最大化なので最大全域木
・グリッドからここに帰着させる考察力は必要。。
"""

import sys
from operator import itemgetter

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
INF = float('inf')
MOD = 10 ** 9 + 7

class UnionFind:
    """ Union-Find木 """

    def __init__(self, n):
        self.n = n
        # 親要素のノード番号を格納。par[x] == xの時そのノードは根
        # 1-indexedのままでOK、その場合は[0]は未使用
        self.par = [i for i in range(n+1)]
        # 木の高さを格納する（初期状態では0）
        self.rank = [0] * (n+1)
        # あるノードを根とする集合に属するノード数
        self.size = [1] * (n+1)
        # あるノードを根とする集合が木かどうか
        self.tree = [True] * (n+1)

    def find(self, x):
        """ 根の検索(グループ番号と言えなくもない) """
        # 根ならその番号を返す
        if self.par[x] == x:
            return x
        else:
            # 走査していく過程で親を書き換える
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    def union(self, x, y):
        """ 併合 """
        # 根を探す
        x = self.find(x)
        y = self.find(y)

        # 木かどうかの判定用
        if x == y:
            self.tree[x] = False
            return
        if not self.tree[x] or not self.tree[y]:
            self.tree[x] = self.tree[y] = False

        # 木の高さを比較し、低いほうから高いほうに辺を張る
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.size[y] += self.size[x]
        else:
            self.par[y] = x
            self.size[x] += self.size[y]
            # 木の高さが同じなら片方を1増やす
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def same(self, x, y):
        """ 同じ集合に属するか判定 """
        return self.find(x) == self.find(y)

    def get_size(self, x):
        """ あるノードの属する集合のノード数 """
        return self.size[self.find(x)]
    
    def is_tree(self, x):
        """ 木かどうかの判定 """
        return self.tree[self.find(x)]

    def len(self):
        """ 集合の数 """
        res = set()
        for i in range(self.n+1):
            res.add(self.find(i))
        # グループ0の分を引いて返却
        return len(res) - 1

H, W = MAP()
sx, sy = MAP()
tx, ty = MAP()
grid = list2d(H, W, 0)
for i in range(H):
    for j, a in enumerate(LIST()):
        grid[i][j] = a

# グリッドのマスを1次元の値にして頂点とする
edges = []
for i in range(H):
    for j in range(W-1):
        cost = grid[i][j] * grid[i][j+1]
        edges.append((i*W+j, i*W+j+1, cost))
for j in range(W):
    for i in range(H-1):
        cost = grid[i][j] * grid[i+1][j]
        edges.append((i*W+j, (i+1)*W+j, cost))

# スコアの最大化なので降順ソート
edges.sort(key=itemgetter(2), reverse=True)
uf = UnionFind(H*W+W)
ans = 0
# ボーナス分
for a, b, cost in edges:
    if not uf.same(a, b):
        uf.union(a, b)
        ans += cost

# マス目に書いてある分
for i in range(H):
    for j in range(W):
        ans += grid[i][j]
print(ans)
