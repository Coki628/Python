# -*- coding: utf-8 -*-

"""
参考：https://codeforces.com/contest/1278/submission/67235649
・試行錯誤の末やっとAC。
・dictやsetで色々やってもダメだった。結局頂点数が多いと、全部見て回るのがつらい。
・参考を元にheapq使ってうまくいった。これだとlogは乗るけど、終わった頂点が減っていくから大丈夫みたい。
　(dictでも終わった頂点は減らすけど、自分より手前の頂点まで全て見に行くからダメ)
・各区間は一意、は問題文読んでたけど、左端右端合わせて一意じゃなくて、
　どれも単独で一意ってことに大分経ってから気づいた。
・それにより、各区間で隣接リスト持ってたのを1つの頂点番号だけ持てば良くなって、
　それまでTLEだったこのheapqのやつが無事ACになった。
"""

import sys
from heapq import heapify, heappush, heappop

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

    def __init__(self, n):
        self.n = n
        self.par = [i for i in range(n+1)]
        self.rank = [0] * (n+1)
        self.size = [1] * (n+1)
        self.tree = [True] * (n+1)

    def find(self, x):
        if self.par[x] == x:
            return x
        else:
            self.par[x] = self.find(self.par[x])
            return self.par[x]

    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            self.tree[x] = False
            return
        if not self.tree[x] or not self.tree[y]:
            self.tree[x] = self.tree[y] = False

        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.size[y] += self.size[x]
        else:
            self.par[y] = x
            self.size[x] += self.size[y]
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def is_same(self, x, y):
        return self.find(x) == self.find(y)

    def get_size(self, x=None):
        if x is not None:
            return self.size[self.find(x)]
        else:
            res = set()
            for i in range(self.n+1):
                res.add(self.find(i))
            return len(res) - 1
    
    def is_tree(self, x):
        return self.tree[self.find(x)]

N = INT()
segs = [0] * (N*2+1)
for i in range(N):
    l, r = MAP()
    segs[l] = i + 1
    segs[r] = -(i + 1)

que = []
edges = []
for i, a in enumerate(segs[1:], 1):
    # 左端
    if a > 0:
        heappush(que, (-i, a))
    # 右端
    else:
        a = -a
        tmp = []
        while que:
            # 自分の左端が来るまで、各頂点を出してくる
            j, b = heappop(que)
            if a == b:
                break
            # キューに戻すので一時保管
            tmp.append((j, b))
            # 辺を追加
            edges.append((a, b))
            # 辺がN個以上で木じゃないのが確定なので即終了
            if len(edges) >= N:
                NO()
                exit()
        for j, b in tmp:
            heappush(que, (j, b))

uf = UnionFind(N)
for a, b in edges:
    uf.union(a, b)

# グラフが1つの木になっているかどうか判定
if uf.get_size() != 1 or not uf.is_tree(1):
    NO()
else:
    YES()
