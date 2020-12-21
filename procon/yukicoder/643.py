"""
・自力AC
・グラフに帰着、整数BFS
・前問と方針は同じ。公式解の数学を全く無視して通した。。
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
EPS = 10 ** -10

def bfs(src):
    """ BFS(一般グラフ、重みなし) """
    from collections import deque

    que = deque([src])
    dist = {}
    dist[src] = 0
    while que:
        ux, uy = que.popleft()
        if abs(ux) > MAX and abs(uy) > MAX:
            continue
        if ux == uy:
            return dist[(ux, uy)]
        vx, vy = uy, ux
        if (vx, vy) not in dist:
            dist[(vx, vy)] = dist[(ux, uy)] + 1
            que.append((vx, vy))
        vx, vy = ux + uy, ux - uy
        if (vx, vy) not in dist:
            dist[(vx, vy)] = dist[(ux, uy)] + 1
            que.append((vx, vy))
    return -1

x, y = MAP()

MAX = 1000
res = bfs((x, y))
print(res)
