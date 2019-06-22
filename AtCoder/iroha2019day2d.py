# -*- coding: utf-8 -*-

"""
・400点自力AC
・最小全域木(ただし高コストな方から構築)
・なんか時間厳しいなーって思ったら、よく見たらエッジ数が40万ある。
・最初TLEして、ソートのキー1つに限定したらギリいけた。
・ちなみにpypyでもほぼ速度変わらず。。
"""

import sys
from operator import itemgetter

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

# Union-Find木
class UnionFind:
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

    # 根の検索(グループ番号と言えなくもない)
    def find(self, x):
        # 根ならその番号を返す
        if self.par[x] == x:
            return x
        else:
            # 走査していく過程で親を書き換える
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    # 併合
    def union(self, x, y):
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

    # 同じ集合に属するか判定
    def same(self, x, y):
        return self.find(x) == self.find(y)

    # あるノードの属する集合のノード数
    def get_size(self, x):
        return self.size[self.find(x)]

    # 木かどうかの判定
    def is_tree(self, x):
        return self.tree[self.find(x)]

    # 集合の数
    def len(self):
        res = set()
        for i in range(self.n+1):
            res.add(self.find(i))
        # グループ0の分を引いて返却
        return len(res) - 1

N,M=MAP()
edges=[None]*M 
for i in range(M):
    a,b,c=MAP()
    edges[i]=(c, a-1, b-1, i+1)

# 最小全域木(クラスカル法)
edges.sort(key=itemgetter(0), reverse=True)
uf=UnionFind(N)
ans=[]
for cost, u, v, idx in edges:
    if not uf.same(u, v):
        uf.union(u, v)
        ans.append(idx)

ans.sort()
[print(a) for a in ans]
