# -*- coding: utf-8 -*-

"""
・自力AC！
・文字を頂点としたグラフとみなしてUnion-Findする。
・結構変わったことした気がしたから、想定解じゃないだろーと思ったら
　意外とほぼ想定解の考え方だったから嬉しい。
"""

import sys
from string import digits, ascii_uppercase
from collections import defaultdict

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

# 文字列と頂点番号のマッピング
tonum = {}
for i, s in enumerate(digits+ascii_uppercase):
    tonum[s] = i

N = INT()
# 最初から頂点番号に変換して持っておく
S = [tonum[s] for s in input()]
T = [tonum[t] for t in input()]

# 同値とみなせる文字をUnionFindで同じ集合に集める
uf = UnionFind(36)
for i in range(N):
    uf.union(S[i], T[i])

# グループ毎にsetに詰める(数字を含むか確認用)
d = defaultdict(set)
for i in range(36):
    root = uf.find(i)
    d[root].add(str(i))

ans = 1
done = [False] * 36
digset = set(digits)
for i in range(N):
    root = uf.find(S[i])
    # 既に計上したグループはスキップ
    if done[root]:
        continue
    done[root] = True
    # 数字が含まれていたら確定なので何もしない
    if len(d[root] & digset):
        continue
    # それ以外は最初だけ0以外1-9の9通り
    if i == 0:
        ans *= 9
    # あとは0-9の10通り
    else:
        ans *= 10
print(ans)
