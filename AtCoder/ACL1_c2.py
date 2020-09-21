"""
・自力AC！これは嬉しい。
・グリッド、最小費用流
・割と辺の張り方が素直で、分かりやすいフロー問が出てくれて助かった。
・ベルマンフォードがTLEしたので、コスト調整してダイクストラ版使った。
・最初、フロー流す量適当にしてたら合わなかった。
　ノー利益の辺で無制限に始点から終点流せばいけると思ったけどダメだった。
・結局、マルの数だけ流すようにしたら合った。
・始点からマスと、マスから終点へ流せる量はしっかり抑えてるのに、なんで任意流量やると合わないんかなー。
・あー分かったかも。多分、任意流量にすると、1回も動けないマスの駒をなかったことにできるから、
　その分そこにスペースがあると判断されて移動距離が長めになってしまってるぽい。
　今回は全駒が確実に使われる(そこに存在する)ものとしてやらないといけないから、
　始点から終点直行の辺は繋いじゃダメなんだね。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
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

class MinCostFlow:
    """ 最小費用流(ダイクストラ版)：O(F*E*logV) """

    INF = 10**18

    def __init__(self, N):
        self.N = N
        self.G = [[] for i in range(N)]
        self.pos = []

    def copy(self):
        res = MinCostFlow(self.N)
        res.G = [[a[:] for a in b] for b in self.G]
        res.pos = [a[:] for a in self.pos]
        return res

    def add_edge(self, fr, to, cap, cost):
        self.pos.append((fr, len(self.G[fr])))
        self.G[fr].append([to, cap, cost, len(self.G[to])])
        self.G[to].append([fr, 0, -cost, len(self.G[fr])-1])

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

    def get_edge(self, i):
        e = self.G[self.pos[i][0]][self.pos[i][1]]
        re = self.G[e[0]][e[3]]
        # from, to, cap, flow, cost
        return (self.pos[i][0], e[0], e[1]+re[1], re[1], e[2])

    def get_edges(self):
        M = len(self.pos)
        res = []
        for i in range(M):
            res.append(self.get_edge(i))
        return res
N, M = MAP()
grid = build_grid(N, M, '#', str)
N += 2
M += 2

cnt = 0
for i in range(1, N):
    for j in range(1, M):
        if grid[i][j] == 'o':
            cnt += 1
MX = cnt
mcf = MinCostFlow(N*M+2)
s = N*M
t = N*M+1
offset = 3000

directions = ((1, 0), (0, 1))
for i in range(1, N):
    for j in range(1, M):
        if grid[i][j] == 'o':
            # 始点 → 各マス
            mcf.add_edge(s, i*M+j, 1, offset)
        if grid[i][j] in ['o', '.']:
            # 各マス → 終点
            mcf.add_edge(i*M+j, t, 1, offset)
            for di, dj in directions:
                ni, nj = i+di, j+dj
                if grid[ni][nj] in ['o', '.']:
                    # マス → マス
                    mcf.add_edge(i*M+j, ni*M+nj, MX, -1)

ans = offset*MX*2 - mcf.flow(s, t, MX)
print(ans)
