"""
参考：ツイート色々
・思った以上にみんなフローで殴ってた。
・始点→Aliceの手→Bobの手→終点、みたいなグラフを構築する。
　後は各手を出せる回数だけ流せるようにして流す。
・これみんなフロー自然と思いつくのかぁ。すごいな。
・これやって思ったのは、Dinicはフローの流量に依存せず流せるので、
　今回みたいに流量になるNが<=10億でも、グラフが小さければ全然流せてしまう、ってこと。
　(今回頂点は8個しかない。)
・なんとなくフローは制約がそこそこ小さい(数1000程度)時じゃないとダメな気がしてたから、
　これは新しい知見だったな。案外フローで殴れる場面ってあるのかもしれない。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class Dinic:
    """ 最大流(Dinic)：O(E*V^2) """

    INF = 10**18

    def __init__(self, n):
        self.n = n
        self.links = [[] for _ in range(n)]
        self.depth = None
        self.progress = None
        self.pos = []

    def copy(self):
        res = Dinic(self.n)
        res.links = [[a[:] for a in b] for b in self.links]
        res.pos = [a[:] for a in self.pos]
        return res

    def add_edge(self, _from, to, cap):
        self.pos.append((_from, len(self.links[_from])))
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

    def flow(self, s, t):
        INF = Dinic.INF
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

    def get_edge(self, i):
        e = self.links[self.pos[i][0]][self.pos[i][1]]
        re = self.links[e[1]][e[2]]
        # from, to, cap, flow
        return (self.pos[i][0], e[1], e[0]+re[0], re[0])

    def get_edges(self):
        M = len(self.pos)
        res = []
        for i in range(M):
            res.append(self.get_edge(i))
        return res

N = INT()
M = 3
A = LIST()
B = LIST()

ans1 = 0
for i in range(M):
    ans1 += min(A[i], B[(i+1)%M])

ans2 = 0
dinic = Dinic(2*M+2)
s = 2*M
t = 2*M + 1
for i in range(M):
    # s to Alice
    dinic.add_edge(s, i, N)
    # Bob to t
    dinic.add_edge(M+i, t, B[i])
    # draw
    dinic.add_edge(i, M+i, A[i])
    # Alice lose
    dinic.add_edge(i, M+(i-1)%M, A[i])
ans2 = N - dinic.flow(s, t)

print(ans2, ans1)
