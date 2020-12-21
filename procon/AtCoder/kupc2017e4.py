# -*- coding: utf-8 -*-

"""
参考：http://www.kupc.jp/static/media/E.8422b5da.pdf
・公式解：集合が木かどうかで処理を分ける。
"""

import sys
from collections import defaultdict

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

class UnionFind:
    def __init__(self, n):
        # 親要素のノード番号を格納。par[x] == xの時そのノードは根
        # 1-indexedのままでOK、その場合は[0]は未使用
        self.par = [i for i in range(n+1)]
        # 木の高さを格納する（初期状態では0）
        self.rank = [0] * (n+1)
        # あるノードを根とする集合に属するノード数
        self.size = [1] * (n+1)

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

N,M=MAP()
V=LIST()
is_tree=[True]*(N+1)

uf=UnionFind(N)
for i in range(M):
    x,y=MAP()
    # 同じ集合でなければ繋ぐ
    if not uf.same(x, y):
        # 今回は木かどうかの判定をしたいので、根が書き換わる前にその情報を入れておく
        if not is_tree[uf.find(x)] or not is_tree[uf.find(y)]:
            is_tree[uf.find(x)]=is_tree[uf.find(y)]=False
        uf.union(x, y)
    # 既に同じ集合なら、そこは木ではなくなる
    else:
        is_tree[uf.find(x)]=False

group=defaultdict(list)
for i in range(N):
    # 集合毎に価値の値をまとめておく
    group[uf.find(i+1)].append(V[i])

ans=0
for k, l in group.items():
    l.sort(reverse=True)
    # その集合が木なら一番小さい値だけ除く
    if is_tree[k]:
        l.pop()
    ans+=sum(l)
print(ans)
