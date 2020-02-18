# -*- coding: utf-8 -*-

"""
参考：https://www.utpc.jp/2012/slides/matchingGame.pdf
・蟻本演習3-5-7
・最小費用流、割当問題亜種
・店、鍵、宝箱の3種類のカテゴリでフローを作る。
・釣り上げコストより多くの鍵を店から買おうとすると、無限に釣り上げて得されてしまうので、それをしない。
・よって、始点 - 店 の容量を、釣り上げて得にならないギリギリ(釣り上げコストと同値)にすると分かれば、
　後は素直にグラフ構築してフロー流せばOK。
・やっぱりpypyはheapqが弱いのか、pythonAC0.7秒、pypyAC0.6秒でほとんど差がなかった。
・ベルマンフォード版で試してみた。メモリ消費が減ったのは想定だったけど、速度も興味深い結果。
　pythonAC1.1秒、pypyAC0.5秒。pypy逆に速くなった。
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

class MinCostFlow:
    """ 最小費用流(ベルマンフォード版、負コストに対応可) """

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
                return -1

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

N, M, D = MAP()
mcf = MinCostFlow(N+M+D+2)
s = N + M + D
t = N + M + D + 1
shops = [[] for i in range(D)]
for i in range(M):
    li = LIST()
    c, shop, k = li[:3]
    shop -= 1
    shops[shop].append((i, c))
    A = [a-1 for a in li[3:]]
    for a in A:
        # 鍵 → 宝箱
        mcf.add_edge(N+i, a, 1, 0)

for i in range(N):
    # 宝箱 → 終点
    mcf.add_edge(i, t, 1, 0)

B = LIST(D)
for i, b in enumerate(B):
    # 始点 → 店
    mcf.add_edge(s, N+M+i, b, 0)
    for key, cost in shops[i]:
        # 店 → 鍵
        mcf.add_edge(N+M+i, N+key, 1, cost)

res = mcf.flow(s, t, N)
print(res)
