# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2020/02/05/071000
・蟻本演習3-5-10
・最小費用流、重み付き二部マッチング、割当問題
・順列をマッチングとして見る。
・(A[i]+A[j]) * abs(i-j) のコストは一見i,jが関連し合っていてどうすればいいか分からなかったけど、
　式変形して、A[i]*abs(i-j) + A[j]*abs(i-j) とすると、iとjの移動コストを独立に考えられる。
・後は完全順列の条件を満たさない移動だけ辺を張らないようにしつつ二部マッチングする。
・計算量は辺が最大約100*100=1万で、F*E*logVは100*1万*約7=700万、pythonAC0.6秒。
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
                for i, (w, cap, cost, _) in enumerate(G[v]):
                    if cap > 0 and dist[w] > dist[v] + cost + H[v] - H[w]:
                        dist[w] = r = dist[v] + cost + H[v] - H[w]
                        prv_v[w] = v; prv_e[w] = i
                        heappush(que, (r, w))
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

N = INT()
A = LIST()

mcf = MinCostFlow(N*2+2)
s = N * 2
t = N * 2 + 1
for i, a in enumerate(A):
    mcf.add_edge(s, i, 1, 0)
    mcf.add_edge(N+i, t, 1, 0)
    for j in range(N):
        # 完全順列なので、A[i] == i のペアには辺を張れない
        if a == j+1:
            continue
        cost = a * abs(i - j)
        mcf.add_edge(i, N+j, 1, cost)

res = mcf.flow(s, t, N)
print(res)
