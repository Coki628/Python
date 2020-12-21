# -*- coding: utf-8 -*-

"""
参考：https://rsk0315.hatenablog.com/entry/2020/05/02/181650
・最小全域木、ダブリング、LCA
・HL分解でも解けるらしいのでまあそのうち。。
→HLDやった。C++を方を参照。
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
        self.par = [i for i in range(n)]
        self.rank = [0] * n
        self.size = [1] * n
        self.tree = [True] * n
        self.grpcnt = n

    def find(self, x):
        """ 根の検索(グループ番号の取得) """
        t = []
        while self.par[x] != x:
            t.append(x)
            x = self.par[x]
        for i in t:
            self.par[i] = x
        return self.par[x]

    def union(self, x, y):
        """ 併合 """
        x = self.find(x)
        y = self.find(y)

        if x == y:
            self.tree[x] = False
            return
        if not self.tree[x] or not self.tree[y]:
            self.tree[x] = self.tree[y] = False

        self.grpcnt -= 1
        if self.rank[x] < self.rank[y]:
            self.par[x] = y
            self.size[y] += self.size[x]
        else:
            self.par[y] = x
            self.size[x] += self.size[y]
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
            return self.grpcnt
    
    def is_tree(self, x):
        """ 木かどうかの判定 """
        return self.tree[self.find(x)]

N, M = MAP()
edges = []
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    if b > a:
        a, b = b, a
    edges.append((c, a, b, i))

# 最小全域木の構築
edges.sort()
mntree = set()
mncost = 0
uf = UnionFind(N)
nodes = [[] for i in range(N)]
root = 0
for c, a, b, _ in edges:
    if not uf.is_same(a, b):
        root = a
        uf.union(a, b)
        mntree.add((a, b))
        nodes[a].append((b, c))
        nodes[b].append((a, c))
        mncost += c

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

MAX = 17
# nxt[k][v] := 頂点vから2^k遡った祖先の頂点
nxt = list2d(MAX, N, -1)
# cost[k][v] := 頂点vから2^k遡った祖先に至るまでの辺の最大コスト
cost = list2d(MAX, N, -1)
depths = [-1] * N
# dfsで各頂点の深さと親を取得
dfs(N, nodes, root)

# ダブリングのテーブル構築
for k in range(1, MAX):
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
    for i in range(MAX):
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
        for i in range(MAX-1, -1, -1):
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

ans = [0] * M
for c, a, b, i in edges:
    if (a, b) in mntree:
        # 最小全域木に含まれるならそのまま
        ans[i] = mncost
    else:
        # 含まれないなら、この辺を足して、a-bパス上で一番重い辺を引く
        ans[i] = mncost + c - get_lca(a, b)
for a in ans:
    print(a)
