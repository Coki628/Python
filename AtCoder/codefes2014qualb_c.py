# -*- coding: utf-8 -*-

"""
・最大流
・絶対想定解じゃないと思うけどちゃんとACした。
　まあ、ちゃんとフローを武器として使えたってことだから良しとするか。
・どういうグラフにフロー流したかは自分ノートの図を参照。
"""

import sys
from collections import Counter

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
    """ 最大流(Dinic)：O(E*V^2) """

    INF = 10 ** 18

    def __init__(self, n):
        self.n = n
        self.links = [[] for _ in range(n)]
        self.depth = None
        self.progress = None

    def copy(self):
        res = Dinic(self.n)
        res.links = [[a[:] for a in b] for b in self.links]
        return res
 
    def add_edge(self, _from, to, cap):
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

S1 = [ord(s)-65 for s in input()]
S2 = [ord(s)-65 for s in input()]
S3 = [ord(s)-65 for s in input()]
N = len(S1) // 2
M = 26

C1 = Counter(S1)
C2 = Counter(S2)
C3 = Counter(S3)

# 最大流
dinic = Dinic(M*3+4)
s = M * 3
s1 = M * 3 + 1
s2 = M * 3 + 2
t = M * 3 + 3

# 始点 -> 始点1
dinic.add_edge(s, s1, N)
# 始点 -> 始点2
dinic.add_edge(s, s2, N)
for i in range(M):
    # 始点1 -> S1
    dinic.add_edge(s1, i, N)
    # 始点2 -> S2
    dinic.add_edge(s2, M+i, N)
    # S1 -> S3
    dinic.add_edge(i, M*2+i, C1[i])
    # S2 -> S3
    dinic.add_edge(M+i, M*2+i, C2[i])
    # S3 -> 終点
    dinic.add_edge(M*2+i, t, C3[i])

# N*2文字分全て流せればOK
res = dinic.flow(s, t)
if res == N * 2:
    YES()
else:
    NO()
