# -*- coding: utf-8 -*-

"""
参考：http://drken1215.hatenablog.com/entry/2019/03/06/102100
　　　http://rippro.org/event/acpc2018/H.pdf
・蟻本演習3-5-8
・Project Selection Problem(燃やす埋める)、最大流、最小カット
・違うグループに属した場合にペナルティを与えるのではなく、
　同じグループに属した場合に利得を与えるように辺を構築する。
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

H, W = MAP()
grid = build_grid(H, W, '', str, space=0)

N = H * W
M = 0
# 黒マスの数
total = 0
# 黒マスが隣り合う箇所の数
adjcnt = 0
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#':
            total += 1
            if i+1 < H and grid[i+1][j] == '#':
                adjcnt += 1
            if j+1 < W and grid[i][j+1] == '#':
                adjcnt += 1

dinic = Dinic(N+adjcnt+2)
s = N + adjcnt
t = N + adjcnt + 1
k = N
for i in range(H):
    for j in range(W):
        # 同じ側から(i, j)と(i+1, j)選択する場合に、利得1を与える
        if i+1 < H and grid[i][j] == grid[i+1][j] == '#':
            dinic.add_link(s, k, 1)
            dinic.add_link(k, i*W+j, INF)
            dinic.add_link(k, (i+1)*W+j, INF)
            k += 1
        if j+1 < W and grid[i][j] == grid[i][j+1] == '#':
            dinic.add_link(k, t, 1)
            dinic.add_link(i*W+j, k, INF)
            dinic.add_link(i*W+j+1, k, INF)
            k += 1

res = dinic.max_flow(s, t)
ans = total - (adjcnt - res)
print(ans)
