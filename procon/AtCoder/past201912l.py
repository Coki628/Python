# -*- coding: utf-8 -*-

"""
・自力AC！
・最小全域木、ビット全探索
・計算量は全部の辺*ビット全探索で(N+M)^2/2*2*5=2万くらい。
　さすがにpython,pypy両方余裕持ってAC。
"""

import sys
from operator import itemgetter
from math import hypot

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
INF = 10 ** 18
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

    def is_same(self, x, y):
        """ 同じ集合に属するか判定 """
        return self.find(x) == self.find(y)

    def get_size(self, x=None):
        if x is not None:
            """ あるノードの属する集合のノード数 """
            return self.size[self.find(x)]
        else:
            """ 集合の数 """
            res = set()
            for i in range(self.n+1):
                res.add(self.find(i))
            # グループ0の分を引いて返却
            return len(res) - 1
    
    def is_tree(self, x):
        """ 木かどうかの判定 """
        return self.tree[self.find(x)]

N, M = MAP()

NM = N + M
XYC = []
for _ in range(N):
    x, y, c = MAP()
    XYC.append((x, y, c, M))
for i in range(M):
    x, y, c = MAP()
    XYC.append((x, y, c, i))

# 全ての塔間の橋を辺として準備する
edges = []
for i in range(NM):
    for j in range(i+1, NM):
        x1, y1, c1, m1 = XYC[i]
        x2, y2, c2, m2 = XYC[j]
        cost = hypot(x1 - x2, y1 - y2)
        if c1 != c2:
            cost *= 10
        edges.append((i, j, cost, 1<<m1|1<<m2))
edges.sort(key=itemgetter(2))

# どの小さい塔を使うかでビット全探索
ans = INF
for S in range(1<<M):
    # UnionFindで最小全域木を構築
    uf = UnionFind(NM)
    sm = 0
    for a, b, cost, m in edges:
        # 辺に使う塔mが使わない集合Sに含まれていたらスキップ
        if S & m:
            continue
        if not uf.is_same(a, b):
            uf.union(a, b)
            sm += cost
    ans = min(ans, sm)
print(ans)
