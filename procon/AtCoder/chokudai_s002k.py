# -*- coding: utf-8 -*-

"""
参考：https://betrue12.hateblo.jp/entry/2019/06/05/213138
・自力ならず。
・グラフに帰着
・各数値を頂点、2値からどちらかを選ぶことを辺として、グラフを構築する。
・すると連結成分毎に、辺が頂点数以上あれば(つまり閉路があれば)、
　その集合の数値を全て選択できることが分かる。
・これにはUnionFindで集合の状態を管理しつつ、木かどうかと集合の要素数も持っておけばいい。
・このグラフの考え方、頭いいなー。気付けるようにしたい。
・計算量は20万頂点へのUnionFindで、pythonAC1.9秒、pypyAC1.0秒。
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

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N = INT()
AB = []
S = set()
for i in range(N):
    a, b = MAP()
    AB.append((a, b))
    S.add(a)
    S.add(b)

# 座標圧縮
zipped, _ = compress(S)
M = len(zipped)
for i in range(N):
    a, b = AB[i]
    AB[i] = (zipped[a], zipped[b])

# とりあえず全ペアを連結
uf = UnionFind(M)
for a, b in AB:
    uf.union(a, b)

# 連結成分毎に見たいので、根を集める
roots = set()
for i in range(M):
    roots.add(uf.find(i))

ans = 0
for root in roots:
    # 連結成分が木なら、どれか1つは使えない
    if uf.is_tree(root):
        ans += uf.get_size(root) - 1
    else:
        ans += uf.get_size(root)
print(ans)
