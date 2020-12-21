# -*- coding: utf-8 -*-

"""
参考：http://www.kupc.jp/static/media/E.67ad1a46.pdf
・蟻本演習3-5-1
・最大流、Dinic、最小カット、グリッド
・マスにコストを持たせるため、各マスをin,outの2つの頂点で表す。(蟻本にも記載があった方法)
・他のマス同士を繋ぐ辺等は切れないので、コストINFで繋ぐ。
・1万頂点、約4万辺で、pythonAC2.0秒ギリギリ、pypyAC1.0秒。
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
HW = H * W
grid = build_grid(H, W, '', str, space=0)

# 最大流：頂点は各マスのinとout、始点、終点
dn = Dinic(2*H*W+2)
s = 2 * H * W
t = 2 * H * W + 1

for i in range(H):
    for j in range(W):
        # 各マス間の辺
        if i != 0:
            dn.add_link(HW+i*W+j, i*W+j-W, INF)
        if i != H-1:
            dn.add_link(HW+i*W+j, i*W+j+W, INF)
        if j != 0:
            dn.add_link(HW+i*W+j, i*W+j-1, INF)
        if j != W-1:
            dn.add_link(HW+i*W+j, i*W+j+1, INF)
        # ヤギがいるマス
        if grid[i][j] == 'X':
            # 始点からヤギマスへの辺
            dn.add_link(s, i*W+j, INF)
            # いるマスは切れないので、in,out間のコストはINF
            dn.add_link(i*W+j, HW+i*W+j, INF)
        else:
            # いないマスは切れるので、in,out間のコストは1
            dn.add_link(i*W+j, HW+i*W+j, 1)
        # グリッド外に出る(終点への)辺
        if i == 0 or i == H-1 or j == 0 or j == W-1:
            dn.add_link(HW+i*W+j, t, INF)

ans = dn.max_flow(s, t)
if ans >= INF:
    print(-1)
else:
    print(ans)
