# -*- coding: utf-8 -*-

"""
・400点自力AC！
・集合の数をライブラリに組み込んだ。
"""

import sys

def input(): return sys.stdin.readline().strip()
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

# 言語毎の連結リストに各人を頂点として入れる
lang=[[] for i in range(M+1)]
for i in range(N):
    for j in LIST()[1:]:
        lang[j].append(i+1)

uf=UnionFind(N)
for i in range(1, M+1):
    for j in range(1, len(lang[i])):
        # 同じ言語で話せる人達を同じ集合に入れていく
        a,b=lang[i][j-1],lang[i][j]
        if not uf.same(a, b):
            uf.union(a, b)

# 最終的に全員が1つの集合に入っていればOK
if uf.len()==1:
    YES()
else:
    NO()
