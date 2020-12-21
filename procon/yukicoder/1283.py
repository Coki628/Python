"""
・しっかり自力AC！
・グリッドダイクストラ
・通行料金を使ったor使ってないのフラグを持たせて、後は普通にダイクストラする。
・500*500=25万グリッドならまだpypyでも間に合うかと思ったらダメだった。。C++で無事AC0.17秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

def dijkstra(grid: list, src: tuple) -> list:
    """ グリッドダイクストラ(H*Wグリッド, 始点(h, w)) """
    from heapq import heappush, heappop

    H, W = len(grid), len(grid[0])
    res = list3d(2, H, W, INF)
    sh, sw = src
    que = [(0, 0, sh, sw)]
    res[0][sh][sw] = 0
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while que:
        dist, used, h, w = heappop(que)
        if res[used][h][w] < dist:
            continue
        for dh, dw in directions:
            nh, nw = h + dh, w + dw
            if grid[nh][nw] == -1:
                continue
            cost = grid[nh][nw]
            if dist + cost < res[used][nh][nw]:
                res[used][nh][nw] = dist + cost
                heappush(que, (dist+cost, used, nh, nw))
            if cost > 1 and not used:
                cost = 1
                if dist + cost < res[1][nh][nw]:
                    res[1][nh][nw] = dist + cost
                    heappush(que, (dist+cost, 1, nh, nw))
    return res

N, M = MAP()
grid = list2d(N+2, N+2, 1)
for i in range(N+2):
    grid[i][0] = -1
    grid[i][N+1] = -1
    grid[0][i] = -1
    grid[N+1][i] = -1
for i in range(M):
    h, w, c = MAP()
    grid[h][w] += c

res = dijkstra(grid, (1, 1))
ans = min(res[0][N][N], res[1][N][N])
print(ans)
