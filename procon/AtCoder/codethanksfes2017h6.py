# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/code-thanks-festival-2017-open/editorial.pdf
　　　https://ferin-tech.hatenablog.com/entry/2017/12/04/233757
　　　https://atcoder.jp/contests/code-thanks-festival-2017-open/submissions/7057697
・蟻本演習2-4-3
・Union-Find、LCA、ダブリング
・公式解1
・今Union-Findの項だけどこれはLCAがメインでUnion−Findは補助的に使う感じ。
・LCAで遡った頂点と一緒に、そこまでの最大コストの情報を保持していくのが肝。
・基本的には通常のLCAの処理の間にコストのmaxに関することを挟み込んでいくんだけど、
　細かい所でバグが取れなくてWAなくすのにすげー苦労した。
・これはやっとうまくいった版。それでもTLEだけどね。。
・その後ちょっと色々最適化したつもり版。なのに速度ほぼ変わらず。。
・他でLCAが速くなったのでこっちも。DFSをスタックに。がしかしあまり変わらず。。
・通った！UnionFindの探索を非再帰にした。pypyTLE→AC1.4秒。これだけで0.5秒以上の改善はでかい。
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

    def find(self, x):
        """ 根の検索(グループ番号と言えなくもない) """
        # 非再帰版
        t = []
        while self.par[x] != x:
            t.append(x)
            x = self.par[x]
        for i in t:
            self.par[i] = x
        return self.par[x]
        # # 根ならその番号を返す
        # if self.par[x] == x:
        #     return x
        # else:
        #     # 走査していく過程で親を書き換える
        #     self.par[x] = self.find(self.par[x])
        #     return self.par[x]

    def union(self, x, y):
        """ 併合 """
        # 根を探す
        x = self.find(x)
        y = self.find(y)
        # 木の高さを比較し、低いほうから高いほうに辺を張る
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
        else:
            self.par[y] = x
            # 木の高さが同じなら片方を1増やす
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1

    def same(self, x, y):
        """ 同じ集合に属するか判定 """
        return self.find(x) == self.find(y)

def dfs(N, nodes, src):
    """ DFS(木、スタック、重みあり) """

    stack = [(src, -1, 0, -1)]
    while stack:
        u, prev, depth, prevc = stack.pop()
        depths[u] = depth
        nxt[0][u] = prev
        # 自分の親への辺のコスト
        cost[0][u] = prevc
        for v, c in nodes[u]:
            if v != prev:
                stack.append((v, u, depth+1, c))

N, M = MAP()
nodes = tuple([] for i in range(N))
uf = UnionFind(N)
# 辺が追加された時系列iを辺のコストとする
for i in range(M):
    a, b = MAP()
    a -= 1
    b -= 1
    if not uf.same(a, b):
        uf.union(a, b)
        # (次の頂点, 辺のコスト)
        nodes[a].append((b, i+1))
        nodes[b].append((a, i+1))

# nxt[k][v] := 頂点vから2^k遡った祖先の頂点
nxt = list2d(17, N, -1)
# cost[k][v] := 頂点vから2^k遡った祖先に至るまでの辺の最大コスト
cost = list2d(17, N, -1)
# 木ではなく森なので、全頂点からやって各木を探索しておく
depths = [-1] * N
for i in range(N):
    if depths[i] == -1:
        # dfsで各頂点の深さと親を取得
        dfs(N, nodes, i)

# ダブリングのテーブル構築
for k in range(1, 17):
    for v in range(N):
        if nxt[k-1][v] == -1:
            # 遡った頂点が根なら、それ以上先はないので-1のまま
            continue
        nxt[k][v] = nxt[k-1][nxt[k-1][v]]
        # ひとつ前と遡った頂点から最大を取る
        cost[k][v] = max(cost[k-1][v], cost[k-1][nxt[k-1][v]])

# LCAの頂点ではなく、そこまでの最大コストを返す
def get_lca(a, b):
    # 深い方をbにする
    if depths[a] > depths[b]:
        a, b = b, a
    # bをaと同じ高さまで持ってくる
    gap = depths[b] - depths[a]
    mx = 0
    for i in range(17):
        # ビットの立っている所を辿れば、好きな高さに移動できる
        if gap>>i & 1:
            # 最大コストの取得
            mx = max(mx, cost[i][b])
            b = nxt[i][b]
    # この時点で一致すればそこがLCA
    if a == b:
        return mx
    else:
        # aとbが一致する直前の高さまで持ってくる
        for i in range(16, -1, -1):
            a2 = nxt[i][a]
            b2 = nxt[i][b]
            if a2 != b2:
                # 最大コストの取得
                mx = max(mx, cost[i][a], cost[i][b])
                a = a2
                b = b2
        # 元々直前の高さだった場合、1度もmaxを取らないのでここで取る
        if a != b:
            mx = max(mx, cost[0][a], cost[0][b])
        return mx

Q = INT()
for i in range(Q):
    x, y = MAP()
    x -= 1
    y -= 1
    if uf.same(x, y):
        print(get_lca(x, y))
    else:
        print(-1)
