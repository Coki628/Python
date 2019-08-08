# -*- coding: utf-8 -*-

"""
・Union-Find祭り。いっぱい作って元のと比較。
・WA臭しかしなかったけどやっぱりWAだった。。
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

N,G,M=MAP()
P=LIST()

edges=[]
for i in range(M):
    a,b=MAP()
    edges.append((a, b))

totalcnt=ans=0
first=True
while totalcnt<G:
    # 現在の辺の状態でUFを構築
    uf=UnionFind(N)
    for i in range(M):
        a,b=edges[i]
        if not uf.same(a, b):
            uf.union(a, b)
    # 誰と繋がってるか確認
    is_valid=[False]*G
    for i, p in enumerate(P):
        if uf.same(0, p):
            is_valid[i]=True
    # 誰も繋がっていない
    if not is_valid.count(True):
        break
    # 最初から繋がっていない分の処理
    if first:
        totalcnt+=is_valid.count(False)
    first=False

    cnts=[None]*M
    for i in range(M):
        # 辺を1つ減らしたUFを構築して元と比較
        uf2=UnionFind(N)
        for j in range(M):
            if i!=j:
                a,b=edges[j]
                if not uf2.same(a, b):
                    uf2.union(a, b)
        is_valid2=[False]*G
        for j, p in enumerate(P):
            if uf2.same(0, p):
                is_valid2[j]=True
        cnts[i]=(i, is_valid.count(True)-is_valid2.count(True))
    # 一番多く繋がりを減らせる辺を消す
    cnts.sort(key=itemgetter(1), reverse=True)
    idx,cnt=cnts[0]
    if cnt==0:
        # 辺を消しても1つも減らせなくなったら1つずつ減らすとする
        ans+=G-totalcnt
        break
    totalcnt+=cnt
    ans+=1
    edges[idx]=(0, 0)
print(ans)
