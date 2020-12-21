"""
参考：https://www.hamayanhamayan.com/entry/2020/06/06/231557
・最小費用流
・フロー、滅多に出ないだけにほんと気づかないなー。
・最大スコアを取りたいので、正負逆にして最小コストを出す。負辺ありなのでベルマンフォード版使う。
・同じ棒を何回使ったかで減点の度合いが変わるので、1つ前との差分を取ったコストで回数分張る。
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

class MinCostFlow:
    """ 最小費用流(ベルマンフォード版、負コストに対応可)：O(F*E*V) """

    INF = 10 ** 18

    def __init__(self, N):
        self.N = N
        self.G = [[] for i in range(N)]

    def add_edge(self, fr, to, cap, cost):
        G = self.G
        G[fr].append([to, cap, cost, len(G[to])])
        G[to].append([fr, 0, -cost, len(G[fr])-1])

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

N, M = MAP()
A = LIST()
B = LIST()
R = LIST()

mcf = MinCostFlow(N+3+2)
s = N + 3
t = N + 3 + 1

for i in range(3):
    # 始点から各ラウンドへの辺
    mcf.add_edge(s, N+i, M, 0)
    for j in range(N):
        # 各ラウンドから各棒への辺
        mcf.add_edge(N+i, j, 1, -(A[j]*B[j]**(i+1)%R[i]))
for i in range(N):
    # 各棒から終点への辺
    mcf.add_edge(i, t, 1, A[i]*B[i])
    mcf.add_edge(i, t, 1, A[i]*B[i]**2 - A[i]*B[i])
    mcf.add_edge(i, t, 1, A[i]*B[i]**3 - A[i]*B[i]**2)

res = mcf.flow(s, t, M*3)
ans = -res
print(ans)
