# -*- coding: utf-8 -*-

"""
参考：https://emtubasa.hateblo.jp/entry/2018/09/25/180008
　　　http://drken1215.hatenablog.com/entry/2018/09/29/155500
・蟻本演習3-5-9
・最大流、辺素パス
・流量は、その2点間を何回通れるか、に対応できる。
・今回は同じ文字に戻ってこれる回数が知りたいので、頂点をin,outに分割してout -> in間にフローを流す。
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

class Dinic:
    """ 最大流(Dinic) """

    INF = 10 ** 18

    def __init__(self, n):
        self.n = n
        self.links = [[] for _ in range(n)]
        self.depth = None
        self.progress = None
 
    def add_link(self, _from, to, cap):
        self.links[_from].append([cap, to, len(self.links[to])])
        self.links[to].append([0, _from, len(self.links[_from]) - 1])
 
    def bfs(self, s):
        from collections import deque

        depth = [-1] * self.n
        depth[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            for cap, to, _ in self.links[v]:
                if cap > 0 and depth[to] < 0:
                    depth[to] = depth[v] + 1
                    q.append(to)
        self.depth = depth
 
    def dfs(self, v, t, flow):
        if v == t:
            return flow
        links_v = self.links[v]
        for i in range(self.progress[v], len(links_v)):
            self.progress[v] = i
            cap, to, rev = link = links_v[i]
            if cap == 0 or self.depth[v] >= self.depth[to]:
                continue
            d = self.dfs(to, t, min(flow, cap))
            if d == 0:
                continue
            link[0] -= d
            self.links[to][rev][0] += d
            return d
        return 0
 
    def max_flow(self, s, t):
        INF = Dinic.INF
        flow = 0
        while True:
            self.bfs(s)
            if self.depth[t] < 0:
                return flow
            self.progress = [0] * self.n
            current_flow = self.dfs(s, t, INF)
            while current_flow > 0:
                flow += current_flow
                current_flow = self.dfs(s, t, INF)

N = INT()
A = [input() for i in range(N)]
M = 26
C = Counter()
outcnt = [0] * M
incnt = [0] * M
for s in A:
    a, b = ord(s[0])-97, ord(s[-1])-97
    # 辺(a, b)の容量、aから出る辺の数、bに入る辺の数
    C[(a, b)] += 1
    outcnt[a] += 1
    incnt[b] += 1

dinic = Dinic(M*2)
for a in range(M):
    # 各文字でin -> out間はINFで繋いでおく
    dinic.add_link(a, M+a, INF)
    for b in range(M):
        # 文字aのout -> 文字bのin
        dinic.add_link(M+a, b, C[(a, b)])

ans = []
for a in range(M):
    # 文字aのoutからinに向かって流す
    res = dinic.max_flow(M+a, a)
    # aで終わる単語が1つ以上あって、aに戻ってこれる流量がaを出る辺の数以上にあればOK
    if incnt[a] >= 1 and res >= outcnt[a]:
        ans.append(chr(a+97))
[print(a) for a in ans]
