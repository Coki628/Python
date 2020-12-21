"""
・自力AC！
・グラフ、木上クエリ、LCA
・ちょっとやってみると、3頂点それぞれの2点間距離を取ると全部の辺2回ずつ通ってると分かる。
・なので、距離3つの総和/2をやる。今回の木は重み付きなので、深さとは別にコストを管理する
　配列を用意して、そっちで距離を取るようにする。2点間距離自体はLCAが取れればO(1)で求まる。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

class LCA:
    """ LCA(最小共通祖先) """

    def __init__(self, nodes, root):
        self.N = len(nodes)
        nv = 1
        MAX = 0
        while nv < self.N:
            nv *= 2
            MAX += 1
        self.MAX = MAX
        self.nxt = list2d(MAX, self.N, -1)
        self.depths, self.cost = self.dfs(self.N, nodes, root)
        for k in range(1, MAX):
            for v in range(self.N):
                if self.nxt[k-1][v] == -1: 
                    continue
                self.nxt[k][v] = self.nxt[k-1][self.nxt[k-1][v]]

    def dfs(self, N, nodes, src):
        stack = [(src, -1)]
        depths = [0] * N
        cost = [0] * N
        while stack:
            u, prev = stack.pop()
            for v, c in nodes[u]:
                if v != prev:
                    self.nxt[0][v] = u
                    depths[v] = depths[u] + 1
                    cost[v] = cost[u] + c
                    stack.append((v, u))
        return depths, cost

    def get_lca(self, a, b):
        if self.depths[a] > self.depths[b]:
            a, b = b, a
        gap = self.depths[b] - self.depths[a]
        for i in range(self.MAX):
            if gap & 1<<i:
                b = self.nxt[i][b]
        if a == b:
            return a
        else:
            for i in range(self.MAX-1, -1, -1):
                a2 = self.nxt[i][a]
                b2 = self.nxt[i][b]
                if a2 != b2:
                    a = a2
                    b = b2
            return self.nxt[0][a]

N = INT()
nodes = [[] for i in range(N)]
for i in range(N-1):
    a, b, c = MAP()
    nodes[a].append((b, c))
    nodes[b].append((a, c))

lca = LCA(nodes, 0)

Q = INT()
for _ in range(Q):
    xyz = LIST()

    ans = 0
    for i in range(3):
        a, b = xyz[i], xyz[(i+1)%3]
        l = lca.get_lca(a, b)
        ans += lca.cost[a] + lca.cost[b] - lca.cost[l]*2
    ans //= 2
    print(ans)
