# -*- coding: utf-8 -*-

"""
参考：https://www.kupc.jp/static/media/I.e20ad30d.pdf
　　　https://competitive12.blogspot.com/2019/11/kupc-2014-i-rain.html
・蟻本演習3-5-13
・最小費用流、需要と供給
・呪文1回につき、雨量がa -> bに1移動する、と考えられる。
・頂点を各地域、辺を呪文(雨量の移動)とする。
・各地域に必要になる雨量を、始点・終点へ向かう辺の容量と対応付ける。
・フローは毎回、具体的なモノとフローに使う頂点や辺を対応付けるのがむずい。
　どうやったらうまく結び付けられるようになるのかね。
・計算量はF*E*logV=15*3万*約13=585万、pythonAC0.5秒、pypyAC0.7秒。
　pypyやっぱheapqが弱いんかな。
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

class MinCostFlow:
    """ 最小費用流(ダイクストラ版)：O(F*E*logV) """

    INF = 10 ** 18

    def __init__(self, N):
        self.N = N
        self.G = [[] for i in range(N)]

    def add_edge(self, fr, to, cap, cost):
        G = self.G
        G[fr].append([to, cap, cost, len(G[to])])
        G[to].append([fr, 0, -cost, len(G[fr])-1])

    def flow(self, s, t, f):
        from heapq import heappush, heappop

        N = self.N; G = self.G
        INF = MinCostFlow.INF

        res = 0
        H = [0] * N
        prv_v = [0] * N
        prv_e = [0] * N

        while f:
            dist = [INF] * N
            dist[s] = 0
            que = [(0, s)]

            while que:
                c, v = heappop(que)
                if dist[v] < c:
                    continue
                for i, (to, cap, cost, _) in enumerate(G[v]):
                    if cap > 0 and dist[to] > dist[v] + cost + H[v] - H[to]:
                        dist[to] = r = dist[v] + cost + H[v] - H[to]
                        prv_v[to] = v; prv_e[to] = i
                        heappush(que, (r, to))
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

N, M, K = MAP()
Qs = [q-1 for q in LIST()]

ABC = []
for i in range(M):
    a, b, c = MAP()
    a -= 1; b -= 1
    ABC.append((a, b, c))

# 事前に行われた呪文の結果を集計
C1 = Counter()
C2 = Counter()
for q in Qs:
    a, b, _ = ABC[q]
    C1[a] += 1
    C2[b] += 1

mcf = MinCostFlow(N+2)
s = N
t = N + 1
# 各呪文の効果に対応する辺を張る
for a, b, c in ABC:
    mcf.add_edge(a, b, INF, c)

for i in range(N):
    # 減った数だけ増やすために始点に繋ぐ
    mcf.add_edge(s, i, C2[i], 0)
    # 増えた数だけ減らすために終点に繋ぐ
    mcf.add_edge(i, t, C1[i], 0)

res = mcf.flow(s, t, K)
if res == INF:
    print(-1)
else:
    print(res)
