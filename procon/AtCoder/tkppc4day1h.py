# -*- coding: utf-8 -*-

"""
・400点自力AC
・ダイクストラ応用
"""

import sys
from heapq import heappop, heappush

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M, K = MAP()
T = [0] * N
for i in range(1, N-1):
    T[i] = INT()
nodes = [[] for i in range(N)]
for i in range(M):
    a, b, c, d = MAP()
    nodes[a-1].append((b-1, c, d))
    nodes[b-1].append((a-1, c, d))

def dijkstra(N: int, nodes: list, src: int) -> list:
    """ ダイクストラ(頂点数, 隣接リスト(0-indexed), 始点) """

    res = [INF] * N
    que = [(0, src)]
    res[src] = 0
    while len(que) != 0:
        dist, cur = heappop(que)
        # 乗り換え時間の調整
        dist += T[cur]
        for nxt, cost, d in nodes[cur]:
            # 待ち時間の調整
            cost += d - dist % d if dist % d else 0
            if dist + cost < res[nxt]:
                res[nxt] = dist + cost
                heappush(que, (dist+cost, nxt))
    return res

res = dijkstra(N, nodes, 0)
if res[N-1] <= K:
    print(res[N-1])
else:
    print(-1)
