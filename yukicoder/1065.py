"""
・さくっと自力AC！
・小数コストのダイクストラ
・多分似たようなやつJOIでやったことある気がする。
"""

import sys
from math import hypot

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

def dijkstra(nodes: list, src: int) -> list:
    """ ダイクストラ(隣接リスト(0-indexed), 始点) """
    from heapq import heappush, heappop

    N = len(nodes)
    res = [INF] * N
    que = [(0, src)]
    res[src] = 0
    while que:
        dist, cur = heappop(que)
        if res[cur] < dist:
            continue
        for nxt, cost in nodes[cur]:
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                heappush(que, (dist+cost, nxt))
    return res

N, M = MAP()
s, t = MAP()
s -= 1; t -= 1

def calc(a, b):
    return hypot(XY[a][0]-XY[b][0], XY[a][1]-XY[b][1])

XY = []
for i in range(N):
    x, y = MAP()
    XY.append((x, y))
nodes = [[] for i in range(N)]
for i in range(M):
    a, b = MAP()
    a -= 1; b -= 1
    c = calc(a, b)
    nodes[a].append((b, c))
    nodes[b].append((a, c))

ans = dijkstra(nodes, s)[t]
print(ans)
