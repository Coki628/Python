# -*- coding: utf-8 -*-

"""
参考：http://kmjp.hatenablog.jp/entry/2013/11/17/1030
・蟻本演習3-5-10
・最小費用流、重み付き二部マッチング、割当問題
・最大量の魔力を集めた中での最小移動距離、が必要。
・魔力と移動距離の条件を両方はフローに含められない。どうすれば、、、
　となってしまったが、最大量の魔力についてはソートして貪欲で求まる。
・その中で、グラフ構築に当たって、同率ボーダーの扱いに工夫が必要で、
　中間点を作って、そこから必要な量だけ終点に流すようにする。
・地味にコーナーケースとして、魔力0の魔法使いは要らないので予め除外する。
"""

import sys
from operator import itemgetter

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

    def copy(self):
        res = MinCostFlow(self.N)
        res.G = [[a[:] for a in b] for b in self.G]
        return res

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
A = []
for i in range(N):
    _, x, y = input().split()
    x, y = int(x), int(y)
    A.append((x, y))
B = []
for i in range(M):
    _, p, x, y = input().split()
    p, x, y = int(p), int(x), int(y)
    B.append((p, x, y))

# 魔法使いを確定と同率ボーダーに分ける
B.sort(key=itemgetter(0), reverse=1)
B2 = []
B3 = []
if M <= N:
    # 魔力0の魔法使いは取りに行く価値がない
    B2 = [b[:] for b in B if b[0] != 0]
else:
    border = B[N-1][0]
    i = 0
    while i < M and B[i][0] >= border and B[i][0] != 0:
        if B[i][0] > border:
            B2.append(B[i])
        else:
            B3.append(B[i])
        i += 1
M2 = len(B2)
M3 = len(B3)

mcf = MinCostFlow(N+M2+M3+3)
s = N + M2 + M3
t = N + M2 + M3 + 1
u = N + M2 + M3 + 2
dist = list2d(N, M2+M3, INF)
for i in range(N):
    x1, y1 = A[i]
    # 始点 -> 各騎士
    mcf.add_edge(s, i, 1, 0)
    for j in range(M2):
        _, x2, y2 = B2[j]
        dist = abs(x1 - x2) + abs(y1 - y2)
        # 各騎士 -> 魔法使い(確定)
        mcf.add_edge(i, N+j, 1, dist)
    for j in range(M3):
        _, x2, y2 = B3[j]
        dist = abs(x1 - x2) + abs(y1 - y2)
        # 各騎士 -> 魔法使い(同率ボーダー)
        mcf.add_edge(i, N+M2+j, 1, dist)

for i in range(M2):
    # 魔法使い(確定) -> 終点
    mcf.add_edge(N+i, u, 1, 0)
for i in range(M3):
    # 魔法使い(同率ボーダー) -> 中間点
    mcf.add_edge(N+M2+i, t, 1, 0)
# 中間点 -> 終点 (M2と合わせてちょうどNになるような量だけ流す)
mcf.add_edge(t, u, N-M2, 0)

res = mcf.flow(s, u, min(N, M2+M3))
print(res)
