"""
・結構さくっと自力AC！
・グリッドBFS、01BFS
・移動方法がノーコストのとコストかかるのがある。
　最初、ダイクストラしようとしたけど、01BFSでいいじゃん、となり、
　グリッドでは多分初の01BFSをやった。
　→初じゃなかった。ARC005Cでやってたわ。
・計算量は3000*3000=900万のグリッドに、ワープの遷移は5*5=25もあるので定数倍もきつい。pypyAC0.93秒。
　pypyだとダイクストラじゃTLEした話もちらほら聞いたから、最初から01BFSに行けたのは正解だったと思う。
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
MOD = 10 ** 19 + 7
EPS = 10 ** -10

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

def bfs(grid, src):
    """ BFS(グリッド、重みなし) """
    from collections import deque

    H, W = len(grid), len(grid[0])
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    dist = list2d(H, W, INF)
    que = deque()
    for h, w in src:
        que.append((h, w))
        dist[h][w] = 0
    while que:
        h, w = que.popleft()
        # 普通に進む
        for dh, dw in directions:
            nh = h + dh
            nw = w + dw
            if grid[nh][nw] == '#':
                continue
            if dist[h][w] < dist[nh][nw]:
                dist[nh][nw] = dist[h][w]
                que.appendleft((nh, nw))
        # ワープする
        for nh in range(h-2, h+3):
            if nh < 0 or nh >= H:
                continue
            for nw in range(w-2, w+3):
                if nw < 0 or nw >= W:
                    continue
                if grid[nh][nw] == '#':
                    continue
                if dist[h][w] + 1 < dist[nh][nw]:
                    dist[nh][nw] = dist[h][w] + 1
                    que.append((nh, nw))
    return dist

H, W = MAP()
sh, sw = MAP()
th, tw = MAP()
grid = build_grid(H, W, '#', str, space=0, padding=1)

res = bfs(grid, [(sh, sw)])
ans = res[th][tw]
if ans == INF:
    print(-1)
else:
    print(ans)
