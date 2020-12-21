# -*- coding: utf-8 -*-

"""
参考：https://competitive12.blogspot.com/2019/08/tdpc-r.html
・蟻本演習3-5-9
・最小費用流、点素パス、強連結成分分解
・頂点を通ったかどうか、が争点になるので、in,out分割して頂点の流量とコストを管理する。
・1回目にはスコアを1追加、2回目以降も通れるけど加点はなし、の条件を、
　in -> outの辺を2本張ることでうまく表現する。(頭いいなー。。)
・今回はコストが利得になるので、コストを負数にしてベルマンフォードの最小費用流を使う。
・負閉路があるとバグるので、予め強連結成分分解で閉路のないグラフに構築し直してからフローを流す。
・まとまった頂点(元閉路)を通ると、その頂点数の分スコアを加点するようにする。
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

def SCC(N, edges):
    """ 強連結成分分解 """

    nodes1 = [[] for i in range(N)]
    nodes2 = [[] for i in range(N)]
    for u, v in edges:
        nodes1[u].append(v)
        nodes2[v].append(u)

    T = []
    visited = [False] * N
    def rec1(cur):
        visited[cur] = True
        for nxt in nodes1[cur]:
            if not visited[nxt]:
                rec1(nxt)
        # 行き止まったところから順にTに入れていく
        T.append(cur)

    # グラフが連結とは限らないので全頂点やる
    for u in range(N):
        if not visited[u]:
            rec1(u)

    visited = [False] * N
    group = [0] * N
    grpnum = 0
    def rec2(cur):
        group[cur] = grpnum
        visited[cur] = True
        for nxt in nodes2[cur]:
            if not visited[nxt]:
                rec2(nxt)

    # 逆順で進めるところまで行く
    for u in reversed(T):
        if not visited[u]:
            rec2(u)
            grpnum += 1
    return grpnum, group

def build_grid(H, W, intv, _type, space=True, padding=False):
    # 入力がスペース区切りかどうか
    if space:
        _input = lambda: input().split()
    else:
        _input = lambda: input()
    _list = lambda: list(map(_type, _input()))
    # 余白の有無
    if padding:
        offset = 1
    else:
        offset = 0
    grid = list2d(H+offset*2, W+offset*2, intv)
    for i in range(offset, H+offset):
        row = _list()
        for j in range(offset, W+offset):
            grid[i][j] = row[j-offset]
    return grid

N = INT()
G = build_grid(N, N, 0, int)
edges = []
for i in range(N):
    for j in range(N):
        if G[i][j]:
            edges.append((i, j))

# 強連結成分分解で閉路を潰す
M, group = SCC(N, edges)
C = Counter(group)
# 分解後のグループを頂点として新しいグラフを構築
G2 = list2d(M, M, 0)
for i in range(N):
    for j in range(N):
        if G[i][j]:
            a, b = group[i], group[j]
            if a != b:
                G2[a][b] = 1

mcf = MinCostFlow(M*2+2)
s = M * 2
t = M * 2 + 1
for i in range(M):
    # 始点 → 各頂点in
    mcf.add_edge(s, i, INF, 0)
    # 各頂点out → 終点
    mcf.add_edge(M+i, t, INF, 0)
    # 各頂点in → out (1回目)
    mcf.add_edge(i, M+i, 1, -C[i])
    # 各頂点in → out (2回目以降)
    mcf.add_edge(i, M+i, INF, 0)
    for j in range(M):
        if G2[i][j]:
            # 頂点i-out → 頂点j-in
            mcf.add_edge(M+i, j, INF, 0)

# 不閉路がなければ正しくフローが流れる
res = mcf.flow(s, t, 2)
print(-res)
