# -*- coding: utf-8 -*-

"""
参考：https://creep06.hatenablog.com/entry/20181018/1539834568
・蟻本演習3-5-10
・最小費用流、重み付き二部マッチング、割当問題
・2つのグループに何をどう割り当てるのかが、なかなか自分だと出ないなー。。
・今回は、同じ幅の積み木は1度しか使えないという条件があるので、
　どの積み木か -> 積み木の幅 での二部マッチングにするとうまくいく。
・codefes2014china_hでも同じことしたけど、全ての積み木を選ぶ必要はないので、
　始点 -> 終点 に直行できる辺も張る。任意流量の最小費用流って言うらしい。
・これもコストでなく利得を最大化したいやつなので、負コスト用にベルマンフォード。
・計算量はF*E*V=1000*4000*2000=80億、、、そしてTLE。。
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

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

N = INT()
S = set()
AB = []
for i in range(N):
    a, b = MAP()
    AB.append((a, b))
    S.add(a)
    S.add(b)

zipped, unzipped = compress(S)
M = len(zipped)
mcf = MinCostFlow(N+M+2)
s = N + M
t = N + M + 1
# 積み木iをどう使うか
for i, (a, b) in enumerate(AB):
    # 始点 -> 各積み木
    mcf.add_edge(s, i, 1, 0)
    # 積み木iを幅a,高さbに使う
    mcf.add_edge(i, N+zipped[a], 1, -b)
    # 積み木iを幅b,高さaに使う
    mcf.add_edge(i, N+zipped[b], 1, -a)

# 積み木を使わない場合の辺
mcf.add_edge(s, t, N, 0)

for i in range(M):
    # ある幅 -> 終点
    mcf.add_edge(N+i, t, 1, 0)

res = mcf.flow(s, t, N)
print(-res)
