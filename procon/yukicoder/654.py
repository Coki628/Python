# -*- coding: utf-8 -*-

"""
参考：https://yukicoder.me/problems/no/654/editorial
・蟻本演習3-5-1
・最大流、Dinic
・座標圧縮した時刻について頂点を拡張する。
・待機する辺は発着があって隣り合う時刻同士にだけ張る。
・1000頂点、約3000辺で、pythonAC1.5秒、pypyAC0.6秒。
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

class Dinic:
    """ 最大流(Dinic) """

    def __init__(self, n):
        self.n = n
        self.links = [[] for _ in range(n)]
        self.depth = None
        self.progress = None
 
    def add_link(self, _from, to, cap):
        self.links[_from].append([cap, to, len(self.links[to])])
        self.links[to].append([0, _from, len(self.links[_from]) - 1])
 
    def bfs(self, s):
        from collections import deque

        depth = [-1] * self.n
        depth[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for cap, to, _ in self.links[v]:
                if cap > 0 and depth[to] < 0:
                    depth[to] = depth[v] + 1
                    q.append(to)
        self.depth = depth
 
    def dfs(self, v, t, flow):
        if v == t:
            return flow
        links_v = self.links[v]
        for i in range(self.progress[v], len(links_v)):
            self.progress[v] = i
            cap, to, rev = link = links_v[i]
            if cap == 0 or self.depth[v] >= self.depth[to]:
                continue
            d = self.dfs(to, t, min(flow, cap))
            if d == 0:
                continue
            link[0] -= d
            self.links[to][rev][0] += d
            return d
        return 0
 
    def max_flow(self, s, t):
        flow = 0
        while True:
            self.bfs(s)
            if self.depth[t] < 0:
                return flow
            self.progress = [0] * self.n
            current_flow = self.dfs(s, t, INF)
            while current_flow > 0:
                flow += current_flow
                current_flow = self.dfs(s, t, INF)

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N, M, D = MAP()
S = set()
edges = []
for i in range(M):
    u, v, p, q, w = MAP()
    q += D
    u -= 1; v -= 1
    S.add(p)
    S.add(q)
    edges.append((u, v, p, q, w))

# 時刻を座標圧縮
zipped, _ = compress(S)
for i, (u, v, p, q, w) in enumerate(edges):
    p = zipped[p]
    q = zipped[q]
    edges[i] = (u, v, p, q, w)

# フライトの辺を張る
L = len(zipped)
dn = Dinic(N * L)
A = [[] for i in range(N)]
for u, v, p, q, w in edges:
    dn.add_link(u*L+p, v*L+q, w)
    # 街毎に発着がある時刻を保持
    A[u].append(p)
    A[v].append(q)

# 各街で待機する辺を張る
for u in range(N):
    A[u].sort()
    for i in range(len(A[u])-1):
        p, q = A[u][i], A[u][i+1]
        # 発着があって隣り合う時刻同士にだけ辺を張る
        dn.add_link(u*L+p, u*L+q, INF)

# 例外処理(最初の街を出発する便、目的の街に到着する便がない)
if not A[0] or not A[N-1]:
    print(0)
    exit()

# 出発と到着の頂点を確認してフローを流す
su = 0
sp = A[0][0]
gv = N - 1
gq = A[N-1][-1]
ans = dn.max_flow(su*L+sp, gv*L+gq)
print(ans)
