# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc014
　　　https://www.creativ.xyz/abc014d-431/
　　　https://hama-du-competitive.hatenablog.com/entry/2016/10/12/003723
・LCA(最小共通祖先)
・オイラーツアー、セグメント木(RMQ)
・セグ木は最小値の値ではなくそのindexを取得
・説明見た時はこっちのがダブリングより分かりやすいと思ったけど、
　実際実装してみたらデータの持たせ方とか案外色々面倒だった。
・そしてpypyでもTLE…やっぱpythonのセグ木は遅いのかな。。
・速度改善版セグ木適用。pypyAC1.6秒！さっすが。
"""

import sys

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

class SegTreeIndex:
    """
    以下のクエリを処理する
    1.update:  i番目の値をxに更新する
    2.query: 区間[l, r)の値とindex(同値があった場合は一番左)を得る
    """
 
    def __init__(self, n, func, init):
        """
        :param n: 要素数(0-indexed)
        :param func: 値の操作に使う関数(min, max)
        :param init: 要素の初期値(単位元)
        """
        self.n = n
        self.func = func
        self.init = init
        # nより大きい2の冪数
        n2 = 1
        while n2 < n:
            n2 <<= 1
        self.n2 = n2
        self.tree = [self.init] * (n2 << 1)
        self.index = [self.init] * (n2 << 1)
        # 1段目(最下段)の初期化
        for i in range(n2):
            self.index[i+n2] = i
        # 2段目以降の初期化
        for i in range(n2-1, -1, -1):
            # 全部左の子の値に更新
            self.index[i] = self.index[i*2]

    def update(self, i, x):
        """
        i番目の値をxに更新
        :param i: index(0-indexed)
        :param x: update value
        """
        i += self.n2
        self.tree[i] = x
        while i > 0:
            i >>= 1
            left, right = i*2, i*2+1
            if self.func(self.tree[left], self.tree[right]) == self.tree[left]:
                self.tree[i] = self.tree[left]
                self.index[i] = self.index[left]
            else:
                self.tree[i] = self.tree[right]
                self.index[i] = self.index[right]
 
    def query(self, a, b):
        """
        [a, b)の値を得る
        :param a: index(0-indexed)
        :param b: index(0-indexed)
        """
        l = a + self.n2
        r = b + self.n2
        s = (self.init, -1)
        while l < r:
            if r & 1:
                r -= 1
                res = self.func(s[0], self.tree[r])
                # 左との一致を優先する
                if res == s[0]:
                    pass
                else:
                    s = (self.tree[r], self.index[r])
            if l & 1:
                res = self.func(self.tree[l], s[0])
                # 左との一致を優先する
                if res == self.tree[l]:
                    s = (self.tree[l], self.index[l])
                else:
                    pass
                l += 1
            l >>= 1
            r >>= 1
        return s

N=INT()
nodes=[[] for i in range(N)]
for i in range(N-1):
    x,y=MAP()
    nodes[x-1].append(y-1)
    nodes[y-1].append(x-1)

# nodes2: オイラーツアーの訪問順に頂点を記録(初めて来た時と別の頂点から帰って来た時)
nodes2=[]
# depths: 頂点uの深さを保持
depths=[-1]*N
# dfsで各頂点の訪問順と深さを取得
def dfs(u, prev, depth):
    nodes2.append((u, depth))
    depths[u]=depth
    for v in nodes[u]:
        if prev!=v:
            dfs(v, u, depth+1)
            nodes2.append((u, depth))
dfs(0, -1, 0)

# sti: オイラーツアーでi番目に訪れた頂点の深さを記録するセグ木
sti=SegTreeIndex(len(nodes2), min, INF)    
# nodes3: 最初に頂点uを訪れたオイラーツアー時のindexを格納
nodes3=[-1]*N
for i in range(len(nodes2)-1, -1, -1):
    u,depth=nodes2[i]
    sti.update(i, depth)
    nodes3[u]=i

Q=INT()
for i in range(Q):
    a,b=MAP()
    # オイラーツアー時の最小index(小さい方をa、大きい方をbにする)
    ai,bi=min(nodes3[a-1], nodes3[b-1]),max(nodes3[a-1], nodes3[b-1])
    idx=sti.query(ai, bi+1)[1]
    # 区間内で一番浅い頂点がlca
    lca=nodes2[idx][0]
    # ab間の距離
    dist=depths[a-1]+depths[b-1]-depths[lca]*2
    print(dist+1)
