# -*- coding: utf-8 -*-

"""
参考：http://abc040.contest.atcoder.jp/data/abc/040/editorial.pdf
・D自力AC！
・UnionFind
・すぐに方針出て、さくっと解けた。
・前にもこういうUnionFindで逆から遡る系はやったことあったしね。
・計算量は多分、年数20万の内ループで道20万人10万で計50万とUnionFindする分
・公式解を参考に、人と道を同じリストに入れてまとめた。
　python1.4秒くらい、まあいくらか速くなったかな。
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

li=[]
N,M=MAP()
for i in range(M):
    a,b,y=MAP()
    li.append((1, y, a, b))

Q=INT()
for i in range(Q):
    v,w=MAP()
    li.append((0, w, i, v))

# 同年に作られた道は通れないので人が先
li.sort(key=itemgetter(0))
# 年代順逆順ソート
li.sort(key=itemgetter(1), reverse=True)

uf=UnionFind(N)
ans=[0]*Q
# 新しい方から遡っていく
for _type, y, a, b in li:
    # 人を確認
    if _type==0:
        ans[a]=uf.get_size(b)
    # 道を復元
    elif _type==1:
        uf.union(a, b)

[print(a) for a in ans]
