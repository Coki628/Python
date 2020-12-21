"""
参考：https://ei1333.github.io/luzhiled/snippets/graph/lowlink.html
・Low-Link作った。実は昔AOJで通したやつでほぼ同じやつ使ってた。
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
EPS = 10 ** -10

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

def low_link(nodes):
    """ Low-Link(関節点aps(set)と橋bridges(list)を列挙する) """

    N = len(nodes)
    visited = [False] * N
    prenum = [0] * N
    parent = [0] * N
    lowest = [0] * N
    bridges = []
    timer = 1

    def rec(cur, prev, timer):
        # curを訪問した直後の処理
        prenum[cur] = lowest[cur] = timer
        timer += 1
        visited[cur] = True
        for nxt in nodes[cur]:
            if not visited[nxt]:
                # curからvへ訪問する直前の処理
                parent[nxt] = cur
                timer = rec(nxt, cur, timer)
                # nxtの探索が終了した直後の処理
                lowest[cur] = min(lowest[cur], lowest[nxt])
                # より近い経路を含まないなら橋とする
                if lowest[nxt] == prenum[nxt]:
                    # 番号の小さい方から入れる
                    bridges.append((min(cur, nxt), max(cur, nxt)))
            elif nxt != prev:
                # cur -> nxt がback-edgeの場合の処理
                lowest[cur] = min(lowest[cur], prenum[nxt])
        return timer
    # 必要な各値の取得(非連結に対応するため全頂点から)
    for i in range(N):
        if not visited[i]:
            timer = rec(i, -1, timer)

    # 間接点
    aps = set()
    # ルートの子ノードの数
    np = 0
    for i in range(1, N):
        p = parent[i]
        if p == 0:
            np += 1
        # 条件2の確認
        elif prenum[p] <= lowest[i]:
            aps.add(p)
    # 条件1の確認
    if np > 1:
        aps.add(0)

    return aps, bridges

N = INT()
edges = []
uf = UnionFind(N)
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b = MAP()
    nodes[a].append(b)
    nodes[b].append(a)
    edges.append((a, b))
    uf.union(a, b)

if uf.get_size() == 1:
    print('Bob')
elif uf.get_size() >= 3:
    print('Alice')
else:
    arts, bridges = low_link(nodes)
    if bridges:
        print('Alice')
    else:
        print('Bob')
