# -*- coding: utf-8 -*-

"""
・蟻本演習2-4-3
・Union-Find応用、二部グラフ、マージテク
・公式解1
・こちらは愚直に色々場合分けする。実装重い。
・そしてどこかバグっていて盛大にWA。時間もオーバーしてた。
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

N, Q = MAP()
uf = UnionFind(N+Q)
nodes = [[] for i in range(N+Q)]
# 二部グラフチェック用の塗り分け
# 0: white, 1: red, 2: blue
color = [0] * (N+Q)
is_bipart = [1] * (N+Q)

# 二部でない状態への更新
def rec(u):
    if visited[u]:
        return
    visited[u] = 1
    color[u] = 0
    is_bipart[u] = 0
    for v in nodes[u]:
        rec(v)

# 集合の色の塗り替え
def rec2(u):
    if visited[u]:
        return
    visited[u] = 1
    if color[u] == 1:
        color[u] += 1
    else:
        color[u] -= 1

for i in range(Q):
    w, x, y, z = MAP()
    x -= 1
    y -= 1
    if w == 1:
        # 両方とも二部グラフではない
        if not is_bipart[x] and not is_bipart[y]:
            # 辺の追加のみ
            pass
        # x側の集合が二部グラフではない
        elif not is_bipart[x]:
            # y側の集合も二部グラフではなくなる
            visited = [0] * (N+Q)
            rec(y)
        # y側の集合が二部グラフではない
        elif not is_bipart[y]:
            # x側の集合も二部グラフではなくなる
            visited = [0] * (N+Q)
            rec(x)
        # 両方とも二部グラフ
        else:
            # 同じ連結成分かどうか
            if uf.same(x, y):
                # 同じ色同士(偶数辺なら違う色同士)を結ぶなら、二部グラフではなくなる
                if z % 2 == 0 and color[x] != color[y] \
                        or z % 2 == 1 and color[x] == color[y]:
                    visited = [0] * (N+Q)
                    rec(x)
            else:
                # 両方とも最初の連結
                if color[x] == 0 and color[y] == 0:
                    if z % 2 == 0:
                        color[x] = 1
                        color[N+i] = 2
                        color[y] = 1
                    else:
                        color[x] = 1
                        color[y] = 2
                # xが未確定
                elif color[x] == 0:
                    # yが赤なら青に
                    if color[y] == 1:
                        # 偶数辺は疑似頂点を挟む
                        if z % 2 == 0:
                            color[N+i] = 2
                            color[x] = 1
                        else:
                            color[x] = 2
                    # yが青なら赤に
                    else:
                        if z % 2 == 0:
                            color[N+i] = 1
                            color[x] = 2
                        else:
                            color[x] = 1
                # yが未確定
                elif color[y] == 0:
                    if color[x] == 1:
                        if z % 2 == 0:
                            color[N+i] = 2
                            color[y] = 1
                        else:
                            color[y] = 2
                    else:
                        if z % 2 == 0:
                            color[N+i] = 1
                            color[y] = 2
                        else:
                            color[y] = 1
                # どちらも最初の連結ではない(色が決まっている)
                else:
                    # 同じ色同士(偶数辺なら違う色同士)を結ぶなら、頂点の塗り替えが必要になる
                    if z % 2 == 0 and color[x] != color[y] \
                            or z % 2 == 1 and color[x] == color[y]:
                        # 要素数の少ない方の集合を塗り替える(ここがマージテクらしい)
                        visited = [0] * (N+Q)
                        if uf.get_size(x) <= uf.get_size(y):
                            rec2(x)
                        else:
                            rec2(y)
        # 辺の連結
        if not uf.same(x, y):
            #  偶数辺を付ける時は間に1頂点挟んでグラフを管理しやすくする
            if z % 2 == 0:
                uf.union(x, N+i)
                uf.union(N+i, y)
                nodes[x].append(N+i)
                nodes[N+1].append(x)
                nodes[y].append(N+1)
                nodes[N+1].append(y)
            else:
                uf.union(x, y)
                nodes[x].append(y)
                nodes[y].append(x)
    else:
        # 連結かどうか
        if uf.same(x, y):
            # 二部グラフではないかどうか
            if not is_bipart[x]:
                YES()
            else:
                # 色が同じかどうか
                if color[x] == color[y]:
                    YES()
                else:
                    NO()
        else:
            NO()
