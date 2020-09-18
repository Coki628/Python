"""
参考：https://atcoder.jp/contests/practice2/submissions/16588371
　　　https://tumoiyorozu.github.io/single-file-ac-library/document_ja/maxflow.html
・最大流、Dinic、復元、グリッド、二部グラフ
・最大流の経路復元、初めてやったな。。
　公式ライブラリを写経して自分のやつにメソッドを追加した。
・とはいえそれがあっても自力ACはできてない。。考察も要る。
・始点側のマスを終点側のマスでグリッドを二部グラフにする。
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
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def build_grid(H, W, intv, _type, space=False, padding=True):
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

class Dinic:
    """ 最大流(Dinic)：O(E*V^2) """

    INF = 10**18

    def __init__(self, n):
        self.n = n
        self.links = [[] for _ in range(n)]
        self.depth = None
        self.progress = None
        self.pos = []

    def copy(self):
        res = Dinic(self.n)
        res.links = [[a[:] for a in b] for b in self.links]
        res.pos = [a[:] for a in self.pos]
        return res

    def add_edge(self, _from, to, cap):
        self.pos.append((_from, len(self.links[_from])))
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

    def flow(self, s, t):
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

    def get_edge(self, i):
        e = self.links[self.pos[i][0]][self.pos[i][1]]
        re = self.links[e[1]][e[2]]
        # from, to, cap, flow
        return (self.pos[i][0], e[1], e[0]+re[0], re[0])

    def get_edges(self):
        M = len(self.pos)
        res = []
        for i in range(M):
            res.append(self.get_edge(i))
        return res

N, M = MAP()
grid = build_grid(N, M, '#', str)
N += 2; M += 2

dinic = Dinic(N*M+2)
s = N*M
t = N*M + 1

directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
for i in range(1, N-1):
    for j in range(1, M-1):
        if grid[i][j] == '#':
            continue
        u = M*i + j
        if (i+j) % 2 == 1:
            # 始点 -> 各マス
            dinic.add_edge(s, u, 1)
            for di, dj in directions:
                ni, nj = i+di, j+dj
                if grid[ni][nj] == '.':
                    v = M*ni + nj
                    # 各マス -> 4方向
                    dinic.add_edge(u, v, 1)
        else:
            # 各マス -> 終点
            dinic.add_edge(u, t, 1)

res = dinic.flow(s, t)
edges = dinic.get_edges()
for u, v, cap, f in edges:
    if u != s and v != t and f:
        ci, cj = divmod(u, M)
        ni, nj = divmod(v, M)
        if ci+1 == ni:
            grid[ci][cj] = 'v'
            grid[ni][nj] = '^'
        elif ci == ni+1:
            grid[ci][cj] = '^'
            grid[ni][nj] = 'v'
        elif cj+1 == nj:
            grid[ci][cj] = '>'
            grid[ni][nj] = '<'
        elif cj == nj+1:
            grid[ci][cj] = '<'
            grid[ni][nj] = '>'

print(res)
for i in range(1, N-1):
    print(''.join(grid[i][1:M-1]))
