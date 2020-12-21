"""
・やっとACした。。デバッグ大変だったー。
・最小費用流、経路復元、任意流量で最小コスト(最大利益)を求める
・辺が1本足りなかった。始点から終点直行の、利益0だけど無制限に通れる辺、みたいなのを追加する。
　これがあると、本来の道を通れないフローはここに流れるので、よしなに最小コストが求まる。
・頂点約50、辺約2500のグラフに、これはベルマンフォード版で、pypyAC0.84秒。
　これでも通るなら負辺の調整やらなくていいから楽だね。
・追記。任意流量の最小費用流、やってた。(AOJ2293)
　蟻本でフロー周り色々やったのに、やっぱり使わないと忘れちゃってるなー。。。
・あと、このグリッドを行列を頂点、マスを辺としてグラフにする形は結構何回も出てきてる。(AOJ2429参照)
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
    """ 最小費用流(ベルマンフォード版、負コストに対応可)：O(F*E*V) """

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
        prv_v = [0] * N
        prv_e = [0] * N

        while f:
            dist = [INF] * N
            dist[s] = 0
            update = True

            while update:
                update = False
                for v in range(N):
                    if dist[v] == INF:
                        continue
                    for i, (to, cap, cost, _) in enumerate(G[v]):
                        if cap > 0 and dist[to] > dist[v] + cost:
                            dist[to] = dist[v] + cost
                            prv_v[to] = v; prv_e[to] = i
                            update = True
            if dist[t] == INF:
                return INF

            d = f; v = t
            while v != s:
                d = min(d, G[prv_v[v]][prv_e[v]][1])
                v = prv_v[v]
            f -= d
            res += d * dist[t]
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

N, K = MAP()
grid = [[] for i in range(N)]
for i in range(N):
    grid[i] = LIST()

mcf = MinCostFlow(N*2+2)
s = N*2
t = N*2 + 1
# 利益なしだけどいくらでも通れる辺を追加して、到達不可にならないようにする
mcf.add_edge(s, t, N*N, 0)
for i in range(N):
    mcf.add_edge(s, i, K, 0)
    mcf.add_edge(N+i, t, K, 0)
    for j in range(N):
        mcf.add_edge(i, N+j, 1, -grid[i][j])

mx = -mcf.flow(s, t, N*N)
ans = [['.' for j in range(N)] for i in range(N)]
edges = mcf.get_edges()
for fr, to, cap, f, c in edges:
    if fr != s and to != t and f:
        h = fr
        w = to - N
        ans[h][w] = 'X'
print(mx)
for i in range(N):
    print(''.join(ans[i]))
