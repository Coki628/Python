"""
・最小費用流
・辺の数に依存しない方のダイクストラ版、これもTLE。。
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
sys.setrecursionlimit(10**9)
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

class MinCostFlow:
    """ 最小費用流(ダイクストラ版2)：O(F*V^2) """

    INF = 10**18

    def __init__(self, N):
        self.N = N
        self.G = [[] for i in range(N)]
        self.pos = []

    def copy(self):
        res = MinCostFlow(self.N)
        res.G = [[a[:] for a in b] for b in self.G]
        res.pos = [a[:] for a in self.pos]
        return res

    def add_edge(self, fr, to, cap, cost):
        self.pos.append((fr, len(self.G[fr])))
        self.G[fr].append([to, cap, cost, len(self.G[to])])
        self.G[to].append([fr, 0, -cost, len(self.G[fr])-1])

    def flow(self, s, t, f):

        N = self.N; G = self.G
        INF = MinCostFlow.INF

        res = 0
        H = [0] * N
        prv_v = [0] * N
        prv_e = [0] * N

        while f:
            dist = [INF] * N
            dist[s] = 0
            used = [False] * N

            while True:
                v = -1
                for u in range(N):
                    if not used[u] and (v == -1 or dist[u] < dist[v]):
                        v = u
                if v == -1:
                    break
                used[v] = True
                for i, (to, cap, cost, _) in enumerate(G[v]):
                    if cap > 0 and dist[to] > dist[v] + cost + H[v] - H[to]:
                        dist[to] = dist[v] + cost + H[v] - H[to]
                        prv_v[to] = v; prv_e[to] = i
            if dist[t] == INF:
                return INF

            for i in range(N):
                H[i] += dist[i]

            d = f; v = t
            while v != s:
                d = min(d, G[prv_v[v]][prv_e[v]][1])
                v = prv_v[v]
            f -= d
            res += d * H[t]
            v = t
            while v != s:
                e = G[prv_v[v]][prv_e[v]]
                e[1] -= d
                G[v][e[3]][1] += d
                v = prv_v[v]
        return res

    def get_edge(self, i):
        e = self.G[self.pos[i][0]][self.pos[i][1]]
        re = self.G[e[0]][e[3]]
        # from, to, cap, flow, cost
        return (self.pos[i][0], e[0], e[1]+re[1], re[1], e[2])

    def get_edges(self):
        M = len(self.pos)
        res = []
        for i in range(M):
            res.append(self.get_edge(i))
        return res

def trisearch_min(lo, hi, func):
    """ 三分探索 """

    while lo+2 < hi:
        m1 = (lo*2+hi) // 3
        m2 = (lo+hi*2) // 3
        res1 = func(m1)
        res2 = func(m2)
        if res1 <= res2:
            hi = m2
        else:
            lo = m1
    return m1, m2

N, K = MAP()
grid = [[] for i in range(N)]
for i in range(N):
    grid[i] = LIST()

offset = 10**12
mcf = MinCostFlow(N*2+2)
s = N*2
t = N*2 + 1
for i in range(N):
    mcf.add_edge(s, i, K, offset)
    mcf.add_edge(N+i, t, K, offset)
    for j in range(N):
        mcf.add_edge(i, N+j, 1, offset-grid[i][j])

def check(x):
    global mcf

    tmp = mcf.copy()
    res = mcf.flow(s, t, x)
    if res != INF:
        res -= offset * x * 3
    mcf = tmp
    return res

m1, m2 = trisearch_min(-1, N*N+1, check)
x = m1 if check(m1) <= check(m2) else m2

mn = mcf.flow(s, t, x)
mn -= offset * x * 3
ans = [['.' for j in range(N)] for i in range(N)]
edges = mcf.get_edges()
for fr, to, cap, f, c in edges:
    if fr != s and to != t and f:
        h = fr
        w = to - N
        ans[h][w] = 'X'
print(-mn)
for i in range(N):
    print(''.join(ans[i]))
