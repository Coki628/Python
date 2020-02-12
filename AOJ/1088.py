# -*- coding: utf-8 -*-

"""
参考：https://nanikaka.hatenadiary.org/entry/20121014/1350186898
　　　https://docs.google.com/presentation/d/104DvdFadmF-MZd7s8y6rBDIw7ka7-Zn8dN13d9cHUUQ/edit#slide=id.ga4628b6_1_396
・蟻本演習3-5-4
・最小費用流(最小費用最大流)
・色々新しい知見を得た問題だった。
・最小費用流、フロー流すとその流量は累積される。
　例えば、1流せるか、2流せるか、って順番に知りたい時は、flowの引数に流量1,2,3ってやっていくんじゃなくて、
　1,1,1ってやっていくのが正解だった。
・時間帯は頂点拡張して、駅*時間帯に一意な番号を振って頂点にする。
・同時到着の流量を1にするため、駅の頂点をin,outに分けてin-out間に流量1の辺を張る。
・同時出発に制限はないので、始点から最初の駅はinじゃなくてoutに、流量無制限で繋ぐ。(ここでハマってた)
・いくつかの大きいテストケースで正しく出るのは確認したけど、提出自体はTLE。
　解説スライドによると計算量500万くらい想定みたいなので、まあしょうがない。
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

def compress(S):
    """ 座標圧縮 """

    zipped, unzipped = {}, {}
    for i, a in enumerate(sorted(S)):
        zipped[a] = i
        unzipped[i] = a
    return zipped, unzipped

while True:
    N = INT()
    if N == 0:
        break
    S = set()
    XYC = [[]] * (N-1)
    for i in range(N-1):
        m = INT()
        XYC[i] = [()] * m
        for j in range(m):
            x, y, c = MAP()
            XYC[i][j] = (x, y, c)
            S.add(x)
            S.add(y)
    g = INT()

    # 時間帯を座標圧縮
    zipped, _ = compress(S)
    for i in range(N-1):
        for j in range(len(XYC[i])):
            x, y, c = XYC[i][j]
            x = zipped[x]
            y = zipped[y]
            XYC[i][j] = (x, y, c)

    M = len(zipped)
    NM = N * M
    s = NM * 2
    t = NM * 2 + 1
    mcf = MinCostFlow(2*NM+2)

    # 始点と終点への辺を張る
    for i in range(M):
        # 始点 → 最初の駅(全時間帯)のout
        mcf.add_edge(s, NM+0*M+i, g, 0)
        # 最後の駅(全時間帯)のout → 終点
        mcf.add_edge(NM+(N-1)*M+i, t, 1, 0)

    # 駅間の辺を張る
    for i in range(N-1):
        fr = i
        to = i + 1
        for j in range(len(XYC[i])):
            x, y, c = XYC[i][j]
            # 駅frのout → 駅toのin
            mcf.add_edge(NM+fr*M+x, to*M+y, 1, c)

    # 各駅(最初の駅以外)のin,outを張る
    for i in range(1, N):
        for j in range(M):
            # 到着時間j以降の全ての時間帯kに乗り継ぎが可能
            for k in range(j, M):
                # 駅iのin → out
                mcf.add_edge(i*M+j, NM+i*M+k, 1, 0)

    # フローを1ずつ流していく
    sm = 0
    ans = (0, sm)
    for i in range(1, g+1):
        res = mcf.flow(s, t, 1)
        if res != -1:
            sm += res
            ans = (i, sm)
        else:
            break
    print(*ans)
