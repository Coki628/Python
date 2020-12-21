# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc004
　　　https://tjkendev.github.io/procon-library/python/min_cost_flow/primal-dual.html
・最小費用流
・クラスはとりあえずそのまま拝借。
・辺の張り方は合ってるはずだけど、pythonTLEのpypyMLE。。
"""

import sys
from heapq import heappush, heappop

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

class MinCostFlow:
    INF = 10**18

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
        H = [0]*N
        prv_v = [0]*N
        prv_e = [0]*N

        while f:
            dist = [INF]*N
            dist[s] = 0
            que = [(0, s)]

            while que:
                c, v = heappop(que)
                if dist[v] < c:
                    continue
                for i, (w, cap, cost, _) in enumerate(G[v]):
                    if cap > 0 and dist[w] > dist[v] + cost + H[v] - H[w]:
                        dist[w] = r = dist[v] + cost + H[v] - H[w]
                        prv_v[w] = v; prv_e[w] = i
                        heappush(que, (r, w))
            if dist[t] == INF:
                return -1

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

R, G, B = MAP()
N = R + G + B

# 最小費用流
mcf = MinCostFlow(1006)
# 始点からRGBの移動前
mcf.add_edge(1001, 1002, R, 0)
mcf.add_edge(1001, 1003, G, 0)
mcf.add_edge(1001, 1004, B, 0)

# 0~1000の頂点番号は座標位置と一致させてある
for v in range(1001):
    # R => 各座標
    mcf.add_edge(1002, v, 1, abs(400-v))
    # G => 各座標
    mcf.add_edge(1003, v, 1, abs(500-v))
    # B => 各座標
    mcf.add_edge(1004, v, 1, abs(600-v))
    # 各座標 => 終点
    mcf.add_edge(v, 1005, 1, 0)

print(mcf.flow(1001, 1005, N))
